import torch
import pytorch_lightning as pl
from . import models
from . import losses
from .dataset import MassachusettsDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import wandb
from loguru import logger

class RoadSegmentationTrainer(pl.LightningModule):
    def __init__(self, options) -> None:
        super().__init__()
        self.options = options
        self.model = getattr(models, options.MODEL.ARCH)(options.MODEL)

        self.loss = getattr(losses, options.LOSS.NAME)(options.LOSS)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.options.OPTIMIZER.LR,
            weight_decay=self.options.OPTIMIZER.WD)

    def log_predictions(self, x, y, y_hat):
        x = x.detach().to('cpu')
        y = y.detach().to('cpu')
        y_hat = y_hat.detach().to('cpu')
        logger.info(f"{x.shape}, {y.shape} (min {y.min()}, max {y.max()}), {y_hat.shape} (min {y_hat.min()}, max {y_hat.max()})")
        imgstack = x[0].permute(1, 2, 0)
        gtstack = y[0]
        predstack = y_hat[0][0]
        for i in range(1, 4):
            imgstack = torch.cat((imgstack, x[i].permute(1, 2, 0)), axis=0)
            gtstack = torch.cat((gtstack, y[i]), axis=0)
            predstack = torch.cat((predstack, y_hat[i][0]), axis=0)
        gtstack = cv2.cvtColor(gtstack.numpy(), cv2.COLOR_GRAY2RGB)
        predstack = cv2.cvtColor(predstack.numpy(), cv2.COLOR_GRAY2RGB)
        imgstack = imgstack.numpy()
        # concat
        gt = np.concatenate((imgstack, gtstack), axis=1)
        pred = np.concatenate((imgstack, predstack), axis=1)
        # wandb
        self.logger.experiment.log({
            "samples": [wandb.Image(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB), caption="predicted"),
                        wandb.Image(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB), caption="gt")]
        })

    def training_step(self, batch, batch_idx):

        # Get data from batch
        x = batch['image']
        y = batch['mask']

        y_hat = self.forward(x)
        loss, loss_dict = self.loss(y_hat, y)

        self.log_dict(loss_dict)   


        if batch_idx % self.options.TRAINING.LOG_FREQ_IMAGES == 0 and self.options.TRAINING.LOG_IMAGES:
            self.log_predictions(x, y, y_hat)
        return {'loss': loss, 'log': loss_dict}

    def validation_step(self, batch, batch_idx):
         # Get data from batch
        x = batch['image']
        y = batch['mask']
        y_hat = self.forward(x)
        loss, loss_dict = self.loss(y_hat, y)
        self.log_dict(loss_dict)
        return {'val_loss': loss, 'log': loss_dict}

    def train_dataloader(self):
        ds = MassachusettsDataset(self.options.DATASET, is_train=True, use_augmentations=True)
        return DataLoader(ds, batch_size=self.options.DATASET.BATCHSIZE, num_workers=self.options.DATASET.WORKERS)

    def val_dataloader(self):
        ds = MassachusettsDataset(self.options.DATASET, is_train=False, use_augmentations=False)
        return DataLoader(ds, batch_size=self.options.DATASET.BATCHSIZE, num_workers=self.options.DATASET.WORKERS)