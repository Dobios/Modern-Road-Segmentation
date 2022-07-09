import torch
import pytorch_lightning as pl
import numpy as np
import cv2
import wandb
from loguru import logger

from . import models
from . import losses
from .metrics import accuracy, f1_score, jaccard
from .utils import patchify, create_submission

class RoadSegmentationTrainer(pl.LightningModule):
    def __init__(self, options) -> None:
        super().__init__()
        self.hparams.update(options)
        self.model = getattr(models, options.MODEL.ARCH)(options.MODEL)

        self.loss = losses.LossLogger(options.LOSS, options.LOSS.NAME)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD)

    def calc_metrics(self, y_hat, y, prefix="val/"):
        patched_y_hat = patchify(y_hat)
        patched_y = patchify(y)
        return {f'{prefix}acc': accuracy(y_hat, y), 
            f'{prefix}patched_acc': accuracy(patched_y_hat, patched_y), 
            f'{prefix}f1': f1_score(y_hat.numpy(), y.numpy()), 
            f'{prefix}patched_f1': f1_score(patched_y_hat.numpy(), patched_y.numpy()),
            f'{prefix}jacc': jaccard(y_hat, y)}

    def log_images(self, x, y, y_hat):
        x = x.detach().to('cpu')
        y = y.detach().to('cpu')
        y_hat = y_hat.detach().to('cpu')
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

        y_hat = self.forward(x).squeeze(1)
        loss, loss_dict = self.loss(y_hat, y)
        for key, value in loss_dict.items():
            self.log(key, value)

        if self.hparams.TRAINING.CALC_METRICS:
            metrics = self.calc_metrics(y_hat.detach().cpu(), y.detach().cpu(), prefix="train/")
            for key, value in metrics.items():
                self.log(key, value, on_epoch=True)

        if batch_idx % self.hparams.TRAINING.LOG_FREQ_IMAGES == 0 and self.hparams.TRAINING.LOG_IMAGES:
            self.log_images(x, y, y_hat)

        return loss

    def validation_step(self, batch, batch_idx):
         # Get data from batch
        x = batch['image']
        y = batch['mask']
        y_hat = self.forward(x).squeeze(1) # only take the first and only channel
        loss, loss_dict = self.loss(y_hat, y)
        self.log("val_loss", loss)
        metrics = self.calc_metrics(y_hat.detach().cpu(), y.detach().cpu())

        for key, value in loss_dict.items():
            self.log("val/" + key, value)
        
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch['image']

        if self.hparams.TEST.USE_TTA:
            y_hat = torch.zeros(x[:, 0, :, :].shape).cpu()
            for x_hat, flip in [(x, False), (x.flip(3), True)]:
                for k in range(0, 4):
                    x_rot = x_hat.rot90(k, (2,3))
                    with torch.no_grad():
                        y_hat_rot = self.forward(x_rot).squeeze(1)
                        y_hat_rot = y_hat_rot.detach().to('cpu').rot90(-k, (1,2))
                    if flip:
                        y_hat_rot = y_hat_rot.flip(2)
                    y_hat += y_hat_rot
            if self.hparams.TEST.TTA_STRATEGY == "mean":
                y_hat /= 8
            elif self.hparams.TEST.TTA_STRATEGY == "thres":
                y_hat = (y_hat > 4).float()
            else:
                raise ValueError("Unknown TTA strategy")
        else:
            with torch.no_grad():
                y_hat = self.forward(x).squeeze(1)
                y_hat = y_hat.detach().to('cpu')

        y_hat_patched = patchify(y_hat)
            
        return {"predictions": y_hat_patched, "mask": y_hat, "names": batch["image_path"]}
    
    def test_epoch_end(self, outputs, dataloader_idx=0):
        """Saves submission file."""
        y_hat = torch.cat([x["predictions"] for x in outputs], dim=0)
        names = []
        for el in outputs:
            names.extend(el["names"])
        create_submission(y_hat, names, self.hparams.TEST.SUBMISSION_PATH)
        return {}
        
