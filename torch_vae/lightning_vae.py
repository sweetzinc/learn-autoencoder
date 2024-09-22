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

class VAELightning(pl.LightningModule):
    def __init__(self,
                 vae_model: nn.Module,
                 params: dict) -> None:
        super().__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        self.save_hyperparameters(ignore=['vae_model'])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        results = self.forward(x, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              kld_weight=self.params['kld_weight'],
                                              batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        results = self.forward(x, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            kld_weight=self.params['kld_weight'],
                                            batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
    
    def on_validation_end(self) -> None:
        self.sample_images()
    
    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        torchvision.utils.save_image(recons.cpu().data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144, device=self.curr_device)
            torchvision.utils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}