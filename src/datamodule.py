from click import option
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from math import ceil, floor
from .datasets import BaseImageDataset, PatchedDataset, load_cil_metadata, load_massachusetts_metadata

class DataModule(pl.LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.options = options
        if ',' in options.TRAIN_DS or ',' in options.VAL_DS or ',' in options.TEST_DS:
            raise NotImplemented("Multiple datasets are not supported yet")

    def load_cil_base(self):
        if 'cil' in self.options.TRAIN_DS and 'cil' in self.options.VAL_DS:
            # CIL needs to be split into train and val
            cil_ds = BaseImageDataset(self.options, load_cil_metadata, is_train=True)
            ratio = self.options.CIL.SPLIT_TEST_VAL_RATIO
            train_size = ceil(len(cil_ds) * ratio[0])
            val_size = len(cil_ds) - train_size
            self.train_ds, self.val_ds = random_split(cil_ds, [train_size, val_size])
        elif 'cil' in self.options.TRAIN_DS:
            self.train_ds = BaseImageDataset(self.options, load_cil_metadata, is_train=True)
        elif 'cil' in self.options.VAL_DS:
            self.val_ds = BaseImageDataset(self.options, load_cil_metadata, is_train=False)

        if 'cil' in self.options.TEST_DS:
            self.test_ds = BaseImageDataset(self.options, load_cil_metadata, is_train=False)

    def load_massachusetts_base(self):
        if 'massachusetts' in self.options.TRAIN_DS:
            self.train_ds = BaseImageDataset(self.options, load_massachusetts_metadata, is_train=True)
        if 'massachusetts' in self.options.VAL_DS:
            self.val_ds = BaseImageDataset(self.options, load_massachusetts_metadata, is_train=False)
        if 'massachusetts' in self.options.TEST_DS:
            self.test_ds = BaseImageDataset(self.options, load_massachusetts_metadata, is_train=False)

    def setup(self, stage=None):
        self.load_cil_base()
        self.load_massachusetts_base()

    def train_dataloader(self):
        return DataLoader(PatchedDataset(self.train_ds, self.options, is_train=True), batch_size=self.options.BATCHSIZE, num_workers=self.options.WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.options.BATCHSIZE, num_workers=self.options.WORKERS)

    def test_dataloader(self):
        # Test is not passed as patched dataset and loaded with batch size 1
        return DataLoader(self.test_ds, batch_size=32)