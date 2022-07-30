import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from loguru import logger

class ConvNeXt(nn.Module):
    def __init__(self, options):
        super(ConvNeXt, self).__init__()
        model = getattr(models, options.VERSION)(pretrained=options.WEIGHTS is not None) #options.WEIGHTS)
        
        self.layer1 = nn.Sequential(model.features[0], model.features[1])
        self.norm1 = model.features[2][0]
        self.up1 = model.features[2][1]

        self.layer2 = model.features[3]
        self.norm2 = model.features[4][0]
        self.up2 = model.features[4][1]

        self.layer3 = model.features[5]
        self.norm3 = model.features[6][0]
        self.up3 = model.features[6][1]

        self.layer4 = model.features[7]

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.norm1(x1)

        x2 = self.up1(x1)
        x2 = self.layer2(x2)
        x2 = self.norm2(x2)

        x3 = self.up2(x2)
        x3 = self.layer3(x3)
        x3 = self.norm3(x3)

        x4 = self.up3(x3)
        x4 = self.layer4(x4)

        return [x1, x2, x3, x4]
