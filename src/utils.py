import cv2
import numpy as np
import torch
import re
from .constants import PATCH_SIZE, CUTOFF
from loguru import logger
from scipy.ndimage import median_filter
from skimage.filters import threshold_local
import matplotlib.pyplot as plt


def read_img(img_fn, resize_to=None):
    """Read image from disk"""
    img = cv2.imread(img_fn)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # crashes  in segfault when using multiple workers
    img = img[...,::-1] # BGR to RGB
    if resize_to is not None and resize_to != (img.shape[1], img.shape[2]):  # resize images
        img = cv2.resize(img, dsize=resize_to)
    return img.copy()

def read_mask(mask_fn, resize_to=None):
    """Read mask from disk"""
    mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
    if resize_to is not None and resize_to != (mask.shape[0], mask.shape[1]):  # resize images
        mask = cv2.resize(mask, dsize=resize_to)
    mask = mask.astype(np.float32)
    if mask.max() > 1:
        mask /= 255
    return mask

def unfold_data(img, mask, stride, size):
    """Unfolds img and mask into a list of patches."""

    img_patches = img.unfold(0, 3, 3).unfold(1, size, stride).unfold(2, size, stride)
    mask_patches = mask.unfold(0, size, stride).unfold(1, size, stride)
    ufold_shape = mask_patches.shape
    img_patches = img_patches.reshape(-1, 3, size, size)
    mask_patches = mask_patches.reshape(-1, size, size)
    return img_patches, mask_patches, ufold_shape

def patchify(mask):
    """Patchify mask and average over patches."""
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = mask.shape[-2] // PATCH_SIZE
    w_patches = mask.shape[-1] // PATCH_SIZE
    patched = mask.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return patched.squeeze(1)

def create_submission(labels, test_filenames, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))

def median_blur_batch(imgs, ksize=15):
    """Median blur batch"""
    batch_size = imgs.shape[0]
    #res = np.empty_like(imgs)
    for i in range(batch_size):
        imgs[i] = median_filter(imgs[i], ksize)
    return imgs

def threshold_batch(imgs, ksize=15):
    """Threshold batch"""
    batch_size = imgs.shape[0]
    #res = np.empty_like(imgs)
    for i in range(batch_size):
        imgs[i] = imgs[i] > threshold_local(imgs[i], ksize)
    return imgs

def remove_low_noise(imgs, thres=0.1):
    imgs[imgs<thres] = 0
    return imgs

def show_first(item):
    """Show img and mask"""
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(item["image"].permute(0, 2, 3, 1)[0])
    plt.subplot(1, 2, 2)
    plt.imshow(item["mask"][0])
    plt.show()