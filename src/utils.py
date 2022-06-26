import cv2
import numpy as np
import torch


def read_img(img_fn):
    """Read image from disk"""
    img = cv2.imread(img_fn)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # crashes  in segfault when using multiple workers
    img = img[...,::-1] # BGR to RGB
    return img.copy()

def read_mask(mask_fn):
    """Read mask from disk"""
    mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask /= 255.0
    return mask

def unfold_data(img, mask, stride, size):
    """Unfolds img and mask into a list of patches."""
    img = torch.tensor(img)
    mask = torch.tensor(mask)
    img_patches = img.unfold(0, size, stride).unfold(1, size, stride).unfold(2, 3, 3)
    mask_patches = mask.unfold(0, size, stride).unfold(1, size, stride)
    img_patches = img_patches.reshape(-1, size, size, 3)
    mask_patches = mask_patches.reshape(-1, size, size)
    return img_patches.numpy(), mask_patches.numpy()
