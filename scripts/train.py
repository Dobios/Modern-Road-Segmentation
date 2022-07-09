import os
import sys
import torch
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


sys.path.append('.')
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


from src.trainer import RoadSegmentationTrainer
from src.datamodule import DataModule
from src.config import get_cfg_defaults

def main(configs, fast_dev_run=False, predict=False):
    log_dir = configs.LOGDIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_properties(device))

    logger.info(f'Hyperparameters: \n {configs}')

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        name=configs.EXP_NAME, 
        project="CIL-RoadSegmentation", 
        entity="jkminder"
        #log_model=True # logs model checkpoints to wandb
    )

    # This is where we initialize the model specific training routines
    # check HPSTrainer to see training, validation, testing loops
    if configs.TRAINING.PRETRAINED_MODEL is not None:
        logger.info(f'Loading pretrained model from {configs.TRAINING.PRETRAINED_MODEL}')
        model = RoadSegmentationTrainer.load_from_checkpoint(configs.TRAINING.PRETRAINED_MODEL, options=configs).to(device)
    else:
        model = RoadSegmentationTrainer(configs).to(device)
    

    # WandB should log gradients and model topology
    wandb_logger.watch(model)

    # this callback saves best 30 checkpoint based on the validation loss
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=10,
        mode='min',
    )

    callbacks = [ckpt_callback]
    
    no_gpus = 1 if torch.cuda.is_available() else 0

    if configs.TRAINING.RESUME_CKPT is not None:
        logger.info(f'Resuming from checkpoint: {configs.TRAINING.RESUME_CKPT}')

    trainer = pl.Trainer(
        gpus=no_gpus,
        logger=wandb_logger,
        max_epochs=configs.TRAINING.EPOCHS, # total number of epochs
        callbacks=callbacks,
        log_every_n_steps=50,
        default_root_dir=log_dir,
        detect_anomaly=True,
        fast_dev_run=fast_dev_run,
        resume_from_checkpoint=configs.TRAINING.RESUME_CKPT,
    )

    datamodule = DataModule(configs.DATASET)
    logger.info('*** Started training ***')
    trainer.fit(model, datamodule=datamodule)
    logger.info('*** Finished training ***')
    logger.info(f'Best validation loss: {ckpt_callback.best_model_score}')
    logger.info(f'Best model saved at {ckpt_callback.best_model_path}')
    if predict:
        logger.info('*** Started testing ***')
        logger.info(f'Loading best model...')
        model = RoadSegmentationTrainer.load_from_checkpoint(ckpt_callback.best_model_path, options=configs).to(device)
        trainer.test(model, datamodule=datamodule)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/baseline.yaml',
                        help='path to the configuration yaml file')
    parser.add_argument('--fdr', action='store_true',
                        help='fast_dev_run mode: will run a full train, val and test loop using 1 batch(es)')
    parser.add_argument('--predict', '-p', action='store_true', default=False)


    args = parser.parse_args()

    # parse configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    main(cfg, args.fdr, args.predict)
