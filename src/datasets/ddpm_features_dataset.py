from multiprocessing.context import assert_spawning
from re import A
import torch
from torch.utils.data import IterableDataset, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from ..utils import read_img, read_mask, unfold_data
import time
from abc import ABC
from itertools import chain
import math
import os
import albumentations as A
from loguru import logger
from copy import copy
from .base_dataset import BaseImageDataset
from .metadata_loaders import load_cil_metadata

class CILFeaturesDataset(BaseImageDataset):
    """Base Class for CIl features dataset. Handles data loading."""
    def __init__(self, options, is_train) -> None:
        super().__init__(options, load_cil_metadata, is_train)

    def __getitem__(self, index):
        start = time.time()
        feats_root = self.options.CIL.PATH + "/features/" + self.metadata.iloc[index]['image_path'].split("/")[-1].split(".")[0]
        img_fn = self.metadata.iloc[index]['image_path']
        item = {'patches': torch.load(feats_root + "_patches.pt"), 'features': torch.load(feats_root + "_feats.pt"), 'masks': torch.load(feats_root + "_masks.pt"), "image_path": img_fn}
        item["statistics"] = {"time-total": time.time()-start}
        return item

class PixelDataset(IterableDataset):
    """Base Class for image dataset. Handles data loading."""
    def __init__(self, options, features_dataset, weighted) -> None:
        super().__init__()
        self.features_dataset = features_dataset
        if len(self.features_dataset) <= 5:
            self.memory = True
        else:
            self.memory = False
        self.shuffle = True
        self.weighted = True

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.features_dataset)       
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.features_dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.features_dataset))
        indices = np.arange(iter_start, iter_end)
        if self.shuffle:
            indices = np.random.permutation(indices)
        if self.memory:
            self.db = {}
            # Load imgs into memory
            for index in indices:
                self.db[index] = self.features_dataset[index]
        for index in indices:
            if self.memory:
                item = self.db[index]
            else:
                item = self.features_dataset[index]
            dims = item["features"].shape[1]
            features = item["features"].permute(1,0,2,3).reshape(dims, -1).permute(1,0)
            y = item["masks"].flatten()
            f_indices = np.arange(features.shape[0])

            if self.weighted:
                no1 = torch.sum(item["masks"])
                wmap = [item["masks"].numel()-no1, no1]
                weights = [wmap[int(v.item())] for v in y]
                for f_index in WeightedRandomSampler(weights, len(f_indices)//2):
                    yield {"features": features[f_index], "y": y[f_index]}
            else:
                if self.shuffle:
                    f_indices = np.random.permutation(f_indices)
                for f_index in f_indices:
                    yield {"features": features[f_index], "y": y[f_index]}
 
