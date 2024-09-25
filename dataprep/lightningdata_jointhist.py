#%%
import os
import numpy as np
import h5py 
from tqdm import tqdm 
from pathlib import Path 
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, dataset_name='dataset'):
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            self.total_samples = len(hdf5_file[dataset_name])
    
    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            data = hdf5_file[self.dataset_name][index]
        return data
    
    def __len__(self):
        # Return the total number of samples
        return self.total_samples
    
#%%

hdf5_path = Path('data.hdf5')
batch_size = 4
loader = DataLoader(HDF5Dataset(hdf5_path), batch_size=batch_size, shuffle=False)

data = np.array(next(iter(loader)))

#%% 
dshape = data.shape[1:]
fig, axes = plt.subplots(1, batch_size, figsize=(20, 5))
for i in range(data.shape[0]):
    print(data[i].shape)
    vmax = 1/(dshape[0]*dshape[1])
    axes[i].imshow(data[i], cmap='magma', vmin=0, vmax=vmax)