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

        # # Get latent vectors and reconstructed images
        # z = self.model.reparameterize(*results[2:])
        # recon = self.model.generate(results[1])
        # # Logging Histograms of Latent Vectors (z)
        # if z is not None:
        #     # Log histogram of latent vectors
        #     self.logger.experiment.add_histogram(
        #         tag='latent_histogram',
        #         values=z.detach().cpu(),
        #         global_step=self.current_epoch
        #     )
        
        # # Logging Histograms of Model Parameters and Gradients
        # for name, param in self.model.named_parameters():
        #     self.logger.experiment.add_histogram(
        #         tag=f'parameters/{name}',
        #         values=param.detach().cpu(),
        #         global_step=self.current_epoch
        #     )
        #     if param.grad is not None:
        #         self.logger.experiment.add_histogram(
        #             tag=f'gradients/{name}',
        #             values=param.grad.detach().cpu(),
        #             global_step=self.current_epoch
        #         )

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

        # Get latent vectors and reconstructed images
        z = self.model.reparameterize(*results[2:])
        recon = self.model.generate(results[1])
        # # Logging Embeddings (Latent Vectors)
        # if batch_idx == 0 and z is not None:
        #     embeddings = z.detach().cpu()
        #     metadata = labels.detach().cpu().tolist()  # Ensure labels are in list format

        #     # Optional: Normalize embeddings if needed
        #     # embeddings = (embeddings - embeddings.mean(dim=0)) / embeddings.std(dim=0)

        #     # Log embeddings to TensorBoard Projector
        #     self.logger.experiment.add_embedding(mat=embeddings, metadata=metadata, 
        #                                          global_step=self.current_epoch, 
        #                                          tag='latent_embeddings')
        
        # Logging Images (Input and Reconstructed) during Validation
        if batch_idx == 0:
            # Log a batch of input images
            input_grid = torchvision.utils.make_grid(x, normalize=True, nrow=16)
            self.logger.experiment.add_image('validation/Input Images',input_grid,self.current_epoch)
            
            # Log a batch of reconstructed images
            recon_grid = torchvision.utils.make_grid(recon, normalize=True, nrow=16)
            self.logger.experiment.add_image('validation/Reconstructed Images',recon_grid,self.current_epoch)
    
    # def on_validation_epoch_end(self) -> None:
    #     # Optionally, sample and log images at the end of the validation epoch
    #     self.sample_images()
    
    def sample_images(self):
        # Sample images from the test dataloader
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        
        # Generate reconstructions
        recons = self.model.generate(test_input, labels=test_label)
        
        # Create grid of reconstructed images
        recon_grid = torchvision.utils.make_grid(recons.cpu().data,normalize=True,nrow=12)

        # Save reconstructed images to disk
        recon_path = os.path.join(self.logger.log_dir, "Reconstructions", f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png")
        torchvision.utils.save_image(recon_grid,recon_path)
        
        # Log reconstructed images to TensorBoard
        self.logger.experiment.add_image('reconstructions/Reconstructions',recon_grid,self.current_epoch)
        
        # Generate new samples from the VAE
        try:
            samples = self.model.sample(144, device=self.curr_device)
            
            # Create grid of sampled images
            samples_grid = torchvision.utils.make_grid(samples.cpu().data, normalize=True,nrow=12)
            
            # Save sampled images to disk
            samples_path = os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_Epoch_{self.current_epoch}.png")
            torchvision.utils.save_image(samples_grid, samples_path)
            
            # Log sampled images to TensorBoard
            self.logger.experiment.add_image('reconstructions/Sampled Images', samples_grid, self.current_epoch)
        except Warning:
            pass  # Handle any warnings if necessary
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}