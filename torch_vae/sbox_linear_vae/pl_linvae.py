#%%
import os 
from pathlib import Path
from typing import List, Callable, Union, Any, TypeVar, Tuple

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class LVAELightning(pl.LightningModule):
    def __init__(self,
                 vae_model: nn.Module,
                 params: dict) -> None:
        super().__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        self.save_hyperparameters(ignore=['vae_model'])
        
        # Ensure that the logger is initialized before using it in sample_images
        # This is handled by PyTorch Lightning after initialization

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        # Forward pass
        results = self.forward(x, labels=labels)
        
        # Compute loss
        train_loss = self.model.loss_function(*results,
                                              kld_weight=self.params['kld_weight'],
                                              batch_idx=batch_idx)
        
        # Log training losses
        self.log_dict(
            {f"train_{key}": val.item() for key, val in train_loss.items()},
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        # Forward pass
        results = self.forward(x, labels=labels)

        # Compute loss
        val_loss = self.model.loss_function(*results,
                                            kld_weight=self.params['kld_weight'],
                                            batch_idx=batch_idx)
        
        # Log validation losses
        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
   
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
