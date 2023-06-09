from .constants import PATCH_SIZE, CUTOFF
from .utils import patchify
from sklearn.metrics import f1_score as f1
import numpy as np
import torch
from torchmetrics.functional import jaccard_index
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def f1_score(y_hat, y):
    # computes F1 score
    f1s = 0.0
    if y_hat.dtype != torch.bool: # patchify returns bool
        y_hat = y_hat.round()
    y_hat = y_hat.astype(np.float32)
    y = y.astype(np.float32)
    #logger.info(f"y_hat: {y_hat.dtype}, y: {y.dtype}, y_hat.shape: {y_hat.shape}, y.shape: {y.shape}")
    for i in range(y_hat.shape[0]):
        f1s += f1(y_hat[i], y[i], average='weighted')
    return f1s/y_hat.shape[0]

def accuracy(y_hat, y):
    # computes classification accuracy
    if y_hat.dtype == torch.bool: # patchify returns bool
        return (y_hat == y).float().mean()
    return (y_hat.round() == y.round()).float().mean()


def jaccard(y_hat, y):
    y_hat =y_hat.round().to(torch.int)
    y = y.round().to(torch.int)
    return jaccard_index(y_hat, y, 2, average="weighted")