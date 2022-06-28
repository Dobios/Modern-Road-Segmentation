import torch 
import torch.nn as nn


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
        return dice, {'dice': dice}  


class BCEDiceLoss(nn.Module):
    def __init__(self, options) -> None:
        super().__init__()
        self.dice = DiceLoss(options)
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice(pred, target)
        return bce + dice, {'bcedice': bce + dice, 'bce': bce, 'dice': dice}
