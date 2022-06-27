import torch
import pytorch_lightning as pl
import models
import losses
import numpy as np
import cv2
import wandb

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
            weight_decay=self.hparams.OPTIMIZER.WD)

    def log_predictions(self, x, y, y_hat):
        imgstack = x[0].permute(1, 2, 0)
        gtstack = y[0]
        predstack = y_hat[0]
        for i in range(1, 4):
            imgstack = torch.cat((imgstack, x[i].permute(1, 2, 0)), axis=0)
            gtstack = torch.cat((gtstack, y[i]), axis=0)
            predstack = torch.cat((predstack, y_hat[i]), axis=0)
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

        self.log_dict(loss_dict, prefix='train_loss')   


        if batch_idx % self.options.TRAINING.LOG_FREQ_IMAGES == 0 and self.options.TRAINING.LOG_IMAGES:
            self.log_predictions(x, y, y_hat)
        return {'loss': loss, 'log': loss_dict}

    def validation_step(self, batch, batch_idx):
         # Get data from batch
        x = batch['image']
        y = batch['mask']
        y_hat = self.forward(x)
        loss, loss_dict = self.loss(y_hat, y)
        self.log_dict(loss_dict, prefix='val_loss')
        return {'val_loss': loss, 'log': loss_dict}