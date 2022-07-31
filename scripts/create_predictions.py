import os
import sys
import torch
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image
sys.path.append('.')
import os
from src.trainer import RoadSegmentationTrainer
from src.datamodule import DataModule
from src.config import get_cfg_defaults

def main(configs, model_path, n, name):
    print(f"Running preds for {name}")
    if not os.path.exists("predictions"):
        os.mkdir("predictions")
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
    model = RoadSegmentationTrainer.load_from_checkpoint(model_path, options=configs).to(device)
    logger.info(f'Loading pretrained model from {model_path}')
    model.eval()
    

    datamodule = DataModule(configs.DATASET)
    datamodule.setup()
    ds = iter(datamodule.val_ds)

    for i in range(n):
        item = next(ds)
        print(item["image_path"])
        item["image"] = item["image"].unsqueeze(0).to(device)
        y = torch.Tensor(model.test_step(item, 0, 0)["mask"])
        save_image(y, f'predictions/{item["image_path"].split("/")[-1].split(".")[0]}_{name}_mask.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/baseline.yaml',
                        help='path to the configuration yaml file')
    parser.add_argument('--model', '-m', type=str, help='path to the model')

    parser.add_argument('-n', help='Number of images from the validation set.', default=10)
    parser.add_argument("--name", help="name of the model")
    args = parser.parse_args()

    # parse configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.TEST.USE_TTA = False # No TTA
    cfg.freeze()
    
    main(cfg, args.model, args.n, args.name)
