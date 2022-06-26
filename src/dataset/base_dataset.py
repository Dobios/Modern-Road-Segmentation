from re import A
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
from ..utils import read_img, read_mask, unfold_data
from abc import ABC
from itertools import chain
import math
import os
import albumentations as A

class BaseDataset(IterableDataset, ABC):
    """
    Base class for datasets. Handles data loading and augmentation.
    """
    def __init__(self, options, is_train, use_augmentations) -> None:
        super().__init__()

        self.options = options
        self.is_train = is_train
        self.use_augmentations = use_augmentations

        # Cache unfolded image
        self.size = self.options.IMG_SIZE
        if self.is_train:
            self.stride = self.options.STRIDE
        else:
            self.stride = self.options.IMG_SIZE

        self.metadata = None # Set in child
        self.path = None # Set in child

        if self.use_augmentations:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(p=options.AUG.RANDOM_BRIGHTNESSCONTRAST_PROB),
                A.Flip(p=options.AUG.FLIP_PROB),
                A.Rotate(limit=options.AUG.ROTATE_LIMIT, p=options.AUG.ROTATE_PROB),
            ])

    
    def create_items(self, patches, mask_patches):
        for patch, mask in zip(patches, mask_patches):
            print(patch.max(), patch.min())

            if self.use_augmentations:
                transformed = self.transform(image=patch, mask=mask)
                patch = transformed['image']
                mask = transformed['mask']
            mask_percentage = mask.sum() / np.prod(mask.shape)
            if mask_percentage < self.options.MIN_MASK_PERCENTAGE:
                continue
            print(patch.max(), patch.min())
            item = {
                'img': torch.Tensor(patch.astype(np.float32)/255.0).permute(2, 0, 1), # normalize to [0, 1] and reshape to C,W,H
                'mask': torch.Tensor(mask)
            }
            yield item

    def load(self, start, end):
        for index in range(start, end):
            img_fn = os.path.join(self.path, self.metadata.iloc[index]['image_path'])
            mask_fn = os.path.join(self.path, self.metadata.iloc[index]['mask_path'])
            x = read_img(img_fn)
            y = read_mask(mask_fn) 
            patches, mask_patches = unfold_data(x, y, self.stride, self.size)
            yield self.create_items(patches, mask_patches)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.metadata)       
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.metadata) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.metadata))
        return chain.from_iterable(self.load(iter_start, iter_end))

class MassachusettsDataset(BaseDataset):
    """
    Dataset for the MASSACHUSETTS dataset
    """
    def __init__(self, options, is_train, use_augmentations) -> None:
        super().__init__(options, is_train, use_augmentations)
        
        metadata = pd.read_csv(os.path.join(self.options.MASSACHUSETTS.PATH, 'metadata.csv'))
        splits = self.options.MASSACHUSETTS.TRAIN_SPLITS if self.is_train else self.options.MASSACHUSETTS.TEST_SPLITS
        metadata = metadata[metadata["split"].isin(splits.split(","))]
        self.metadata = metadata[["tiff_image_path", "tif_label_path"]]
        self.metadata = self.metadata.reset_index(drop=True)
        self.metadata.columns = ["image_path", "mask_path"]

        self.path = self.options.MASSACHUSETTS.PATH