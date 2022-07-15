from turtle import back
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from loguru import logger
from . import backbones

"""
Implementation of UPerNet from:
https://openaccess.thecvf.com/content_ECCV_2018/papers/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.pdf

Implementation adapted from:
https://sh-tsang.medium.com/review-upernet-unified-perceptual-parsing-for-scene-understanding-4d03247ddad2
https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/uper_head.py#L13
"""

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output



def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=128):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1) for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, options, num_classes=1, backbone='convnext_base', pretrained=True, **_):
        super(UperNet, self).__init__()
        self.options = options

        if backbone == 'convnext_base':
            feature_channels = [128, 256, 512, 1024]
        elif backbone == 'convnext_small' or backbone == 'convnext_tiny':
            feature_channels = [96, 192, 384, 768]
        elif backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
        else:
            feature_channels = [256, 512, 1024, 2048]
        
        self.backbone = getattr(backbones, options.BACKBONE)(backbone=backbone, pretrained=pretrained)
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        x = F.sigmoid(x)
        #logger.info(x.shape)
        return x

