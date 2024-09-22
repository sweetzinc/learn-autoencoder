#%%
import os 
from pathlib import Path
from typing import List, Callable, Union, Any, TypeVar, Tuple
from matplotlib import pyplot as plt 

# PyTorch
import torch
from torch.utils.data import random_split, DataLoader

# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size:int=128):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

#%% 
if __name__ == "__main__" : 
    DATASET_PATH = "/mounted_data/downloaded"

    mnist_datamodule = MNISTDataModule(data_dir=DATASET_PATH, batch_size=4)
    mnist_datamodule.setup('test')
#%%
    x, labels = next(iter(mnist_datamodule.test_dataloader()))

    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([x[0], x[1]], dim=0), 
                                       nrow=2, normalize=True, value_range =(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,2))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
# %%
