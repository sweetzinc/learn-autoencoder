#%%
import os
import numpy as np
import h5py 
from tqdm import tqdm 
from pathlib import Path 
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, ds_key='images'):
        self.h5_file_path = h5_file_path
        self.file = None  # Will be initialized in __getitem__
        self.ds_key = ds_key
        with h5py.File(self.h5_file_path, 'r') as f:
            self.dataset_length = len(f[self.ds_key]) 

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Lazy initialization of the file handle
        if self.file is None:
            self.file = h5py.File(self.h5_file_path, 'r')
        
        data = self.file[self.ds_key][idx]
        data = torch.tensor(data).unsqueeze(0)
        labels = torch.tensor(0)  # Dummy label
        return data, labels

    def __del__(self):
        if self.file is not None:
            self.file.close()


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Ensure each worker has its own file handle
    dataset.file = h5py.File(dataset.h5_file_path, 'r')


class HDF5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_file_path,
        batch_size=32,
        num_workers=4,
        val_split=0.2,
        test_split=0.1,
        seed=42
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Called on every GPU/process
        if not self.dataset:
            self.dataset = HDF5Dataset(self.h5_file_path)
        
        dataset_length = len(self.dataset)
        val_size = int(dataset_length * self.val_split)
        test_size = int(dataset_length * self.test_split)
        train_size = dataset_length - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn
        )

    @staticmethod
    def worker_init_fn(worker_id):
        # Ensure each worker has its own file handle
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        # Handle Subset datasets
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        if hasattr(dataset, 'file') and dataset.file is not None:
            dataset.file.close()
            dataset.file = None
#%%
if __name__ == "__main__" : 
    if os.name == 'nt':
        data_dir = r'C:\docker_share\asn_workinprogress'
    else:
        data_dir = '/mounted_data/asn_workinprogress'


    dataset = HDF5Dataset(Path(data_dir,'images_A.h5'), ds_key='images')
    batch_size = 4
    loader = DataLoader(dataset,batch_size=batch_size,
                        shuffle=True,num_workers=4,worker_init_fn=worker_init_fn)

    for data, labels in loader:
        print(data.shape)
        print(labels)
        break

    dshape = data.shape[1:]
    fig, axes = plt.subplots(1, batch_size, figsize=(20, 5))
    for i in range(data.shape[0]):
        print(data[i].shape)
        axes[i].imshow(data[i], cmap='magma', vmin=0, vmax=255)
# %%
if __name__ == "__main__" : 
    data_config = { 'h5_file_path':'/mounted_data/asn_workinprogress/images_A.h5', 
                   'batch_size':16 }
    datamodule = HDF5DataModule(h5_file_path=data_config['h5_file_path'], 
                                      batch_size=data_config['batch_size'],
                                      num_workers=0)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    for data, labels in train_loader:
        print("data.shape=", data.shape)
        print("labels=", labels)
        break
# %%
