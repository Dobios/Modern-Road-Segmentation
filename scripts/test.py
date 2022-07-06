import os
import sys
import torch
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


sys.path.append('.')



from src.trainer import RoadSegmentationTrainer
from src.datamodule import DataModule
from src.config import get_cfg_defaults

def main(configs, model_path):
    log_dir = configs.LOGDIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.add(
        os.path.join(log_dir, 'test.log'),
        level='INFO',
        colorize=False,
    )

    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_properties(device))


    # This is where we initialize the model specific training routines
    # check HPSTrainer to see training, validation, testing loops
    model = RoadSegmentationTrainer(configs).to(device)
    logger.info(f'Loading pretrained model from {model_path}')
    model.load_state_dict(model_path)
    
    
    no_gpus = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        gpus=no_gpus,
        detect_anomaly=True,
    )

    datamodule = DataModule(configs.DATASET)
  
    logger.info('*** Started testing ***')
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/baseline.yaml',
                        help='path to the configuration yaml file')
    parser.add_argument('--model', '-m', type=str, help='path to the model')



    args = parser.parse_args()

    # parse configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    main(cfg, args.model)
