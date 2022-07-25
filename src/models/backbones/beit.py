from transformers import BEIT_PRETRAINED_MODEL_ARCHIVE_LIST, BeitModel
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation

import torch.nn as nn
from mmcv.cnn import build_norm_layer

class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.
    A neck structure connect ViT backbone and decoder_heads.
    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).

    Code from: https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/necks/featurepyramid.py
    """

    def __init__(self,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(Feature2Pyramid, self).__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)


class BEiT(nn.Module):
    def __init__(self, options):
        super(BEiT, self).__init__()
        # self.model = BeitModel.from_pretrained(BEIT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

        # self.feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
        self.model = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-large-patch16-224')

        self.neck = Feature2Pyramid(embed_dim=768, rescales=[4, 2, 1, 0.5])

    def forward(self, x):
        return self.model(x)
        # x1 = self.encoder.layer[0](x0)[0]
        # x2 = self.encoder.layer[1](x1)[0]
        # x3 = self.encoder.layer[2](x2)[0]
        # x4 = self.encoder.layer[3](x3)[0]
        # x5 = self.encoder.layer[4](x4)[0]
        # x6 = self.encoder.layer[5](x5)[0]
        # x7 = self.encoder.layer[6](x6)[0]
        # x8 = self.encoder.layer[7](x7)[0]
        # x9 = self.encoder.layer[8](x8)[0]
        # x10 = self.encoder.layer[9](x9)[0]
        # x11 = self.encoder.layer[10](x10)[0]

        # print("x0: " + str(x0.shape))
        # print("x1: " + str(x1.shape))
        # print("x2: " + str(x2.shape))
        # print("x3: " + str(x3.shape))
        # print("x4: " + str(x4.shape))
        # print("x5: " + str(x5.shape))
        # print("x6: " + str(x6.shape))
        # print("x7: " + str(x7.shape))
        # print("x8: " + str(x8.shape))
        # print("x9: " + str(x9.shape))
        # print("x10: " + str(x10.shape))
        # print("x11: " + str(x11.shape))

        # return [self.neck([x3, x5, x7, x11])]
