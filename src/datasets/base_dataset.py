from multiprocessing.context import assert_spawning
from re import A
import torch
from torch.utils.data import IterableDataset, Dataset
import pandas as pd
import numpy as np
from ..utils import read_img, read_mask, unfold_data
from abc import ABC
from itertools import chain
import math
import os
import albumentations as A
from loguru import logger

class BaseImageDataset(Dataset):
    """Base Class for image dataset. Handles data loading."""
    def __init__(self, options, metadata_loader, is_train) -> None:
        super().__init__()
        self.options = options
        self.metadata = metadata_loader(options, is_train)
        self.has_mask = "mask_path" in self.metadata.columns
        self.cut_size = self.options.CUT_SIZE
        self.is_train = is_train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_fn = self.metadata.iloc[index]['image_path']
        img = read_img(img_fn)
        if self.cut_size is not None:
            img = img[:self.cut_size, :self.cut_size, :]
        item = {'image': torch.Tensor(img.astype(np.float32)/255.0).permute(2, 0, 1), "image_path": img_fn}
        if self.has_mask:
            mask_fn = self.metadata.iloc[index]['mask_path']
            mask = read_mask(mask_fn)
            if self.cut_size is not None:
                mask = mask[:self.cut_size, :self.cut_size]
            item['mask'] = torch.Tensor(mask)
        return item

class PatchedDataset(IterableDataset, ABC):
    """
    Wraps a BaseImageDataset with image and mask and converts it into smaller patches. Handles unfolding and augmentation.
    """
    def __init__(self, image_dataset: BaseImageDataset, options, is_train) -> None:
        super().__init__()

        self.dataset = image_dataset
        # assert(self.dataset.has_mask) # the wrapped dataset must have a mask
        self.options = options
        self.is_train = is_train
        self.use_augmentations = self.options.USE_AUG and self.is_train

        # Cache unfolded image
        self.size = self.options.PATCH_SIZE
        if self.is_train:
            self.stride = self.options.PATCH_STRIDE
        else:
            self.stride = self.options.IMG_SIZE

        self.shuffle = self.options.SHUFFLE if self.is_train else False

        if self.use_augmentations:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(p=options.AUG.RANDOM_BRIGHTNESSCONTRAST_PROB),
                A.Flip(p=options.AUG.FLIP_PROB),
                A.Rotate(limit=options.AUG.ROTATE_LIMIT, p=options.AUG.ROTATE_PROB),
            ])

    
    def create_items(self, patches, mask_patches):
        indices = np.arange(patches.shape[0])
        if self.shuffle:
            indices = np.random.permutation(indices)
        for index in indices:
            patch = patches[index] # (H, W, C) to (C, H, W) as augmentation require this
            mask = mask_patches[index]
            if self.use_augmentations:
                transformed = self.transform(image=patch.permute(1, 2, 0).numpy(), mask=mask.numpy())
                patch = torch.Tensor(transformed['image'].astype(np.float32)).permute(2,0,1)
                mask = torch.Tensor(transformed['mask'])
            mask_percentage = mask.sum() / np.prod(mask.shape)
            if mask_percentage < self.options.MIN_MASK_PERCENTAGE:
                continue
            item = {
                'image': patch,
                'mask': mask
            }
            yield item

    def load(self, start, end):
        indices = np.arange(start, end)
        if self.shuffle:
            indices = np.random.permutation(indices)
        for index in indices:
            item = self.dataset[index]
            patches, mask_patches,_ = unfold_data(item["image"], item["mask"], self.stride, self.size)
            yield self.create_items(patches, mask_patches)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.dataset)       
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
        return chain.from_iterable(self.load(iter_start, iter_end))
