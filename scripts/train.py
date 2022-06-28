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
from src.config import get_cfg_defaults

def main(configs, fast_dev_run=False):
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
    model = RoadSegmentationTrainer(configs).to(device)
    
    # WandB should log gradients and model topology
    wandb_logger.watch(model)

    # this callback saves best 30 checkpoint based on the validation loss
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=10, # reduce this if you don't have enough storage
        mode='min',
    )

    callbacks = [ckpt_callback]
    
    no_gpus = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        gpus=no_gpus,
        logger=wandb_logger,
        max_epochs=configs.TRAINING.EPOCHS, # total number of epochs
        callbacks=callbacks,
        log_every_n_steps=50,
        terminate_on_nan=True,
        default_root_dir=log_dir,
        fast_dev_run=fast_dev_run,
    )


    logger.info('*** Started training ***')
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/baseline.yaml',
                        help='path to the configuration yaml file')
    parser.add_argument('--fdr', action='store_true',
                        help='fast_dev_run mode: will run a full train, val and test loop using 1 batch(es)')

    args = parser.parse_args()

    # parse configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    main(cfg, args.fdr)