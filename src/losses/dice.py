from turtle import forward
import torch 
import torch.nn as nn

loss_short_names = {
    "DiceLoss": "dice",
    "BCEDiceLoss": "bcedice",
    "BCELoss": "bce",
}

class LossLogger(nn.Module):
    """Logs all available losses but returns only the selected one."""
    def __init__(self, options, loss_fun) -> None:
        super().__init__()
        self.options = options

        self.losses = {
            'loss/dice': DiceLoss(options),
            'loss/bce': BCELoss(options),
            'loss/bcedice': BCEDiceLoss(options),
        }
        self.loss_fun = "loss/" + loss_short_names[loss_fun]

    def forward(self, pred, gt):
        losses = {}
        for key, value in self.losses.items():
            losses[key] = value(pred, gt)[0]
        return losses[self.loss_fun], losses

class DiceLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.smooth = options.DICE.SMOOTH

    def forward(self, pred, target):
        """Copied from https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388"""
        num = target.size(0)
        pred = pred.reshape(num, -1)
        target = target.reshape(num, -1)
        intersection = (pred * target)
        dice = (2. * intersection.sum(1) + self.smooth) / (pred.sum(1) + target.sum(1) + self.smooth)
        dice = 1 - dice.sum() / num
        return dice, {'loss/dice': dice}  


class BCEDiceLoss(nn.Module):
    def __init__(self, options) -> None:
        super().__init__()
        self.dice = DiceLoss(options)
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice(pred, target)
        return bce + dice[0], {'loss/bcedice': bce + dice[0], 'loss/bce': bce, 'loss/dice': dice[0]}

class BCELoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        return bce, {'loss/bce': bce}