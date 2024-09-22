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
import torchvision.utils as tvutils

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class VanillaVAE(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 in_channels: int = 3,
                 hidden_dims: List = None,
                 width: int = 32,
                 height: int = 32,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.example_input_array = torch.zeros(3, in_channels, width, height)
        self.in_channels = in_channels
        # Encoder
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.hidden_dims = hidden_dims
        self.num_decoder_layers = len(hidden_dims) 

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        encshape = self.encoder(self.example_input_array).shape 
        self.encshape = encshape

        self.fc_mu = nn.Linear(hidden_dims[-1]*encshape[2]*encshape[3], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*encshape[2]*encshape[3], latent_dim)


        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1]*encshape[2]*encshape[3])
        
        # Build Decoder
        modules = []
        for i in range(self.num_decoder_layers - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i - 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[0],
                                               hidden_dims[0],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[0]),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[0], out_channels=self.in_channels,
                                      kernel_size=3, padding=1),
                            nn.Sigmoid())

    def encode(self, x: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], *self.encshape[2:])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x: torch.tensor, **kwargs) -> Tuple[torch.tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), x, mu, log_var
    
    def loss_function(self, recons, x, mu, log_var, kld_weight=0.005, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def sample(self, num_samples:int, device:str, **kwargs) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples
    
    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


#%% Set up vanilla vae 
vae_config = {
    'in_channels': 1,
    'latent_dim': 2,
    'hidden_dims': [32, 64, ],
    'width': 28, 'height': 28}

vae_model = VanillaVAE(**vae_config)
#%% Set up LightningModule
lightning_config = {
    'LR': 0.005,
    'weight_decay': 0.0,
    'scheduler_gamma': 0.95,
    'kld_weight': 0.00025,
    'manual_seed': 1265 }

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
        tvutils.save_image(recons.cpu().data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144, device=self.curr_device)
            tvutils.save_image(samples.cpu().data,
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

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
pl.seed_everything(lightning_config['manual_seed'], True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

lightning_module = VAELightning(vae_model, lightning_config)

#%%
DATASET_PATH = "/mounted_data/downloaded"
CHECKPOINT_PATH = "/workspace/torch_vae/saved_models/vanilla_pl"

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

mnist_datamodule = MNISTDataModule(data_dir=DATASET_PATH)
mnist_datamodule.setup('fit')
mnist_datamodule.setup('test')
#%%

if __name__ == '__main__' :
    logging_params= {'save_dir': '/workspace/torch_vae/logs/', 'name': 'VanillaVAE'}

    tb_logger = TensorBoardLogger(save_dir=logging_params['save_dir'],
                                name=logging_params['name'],)

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor("epoch"),
            ModelCheckpoint(save_top_k=2,
                            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                            monitor="val_loss",
                            save_last=True),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10,)

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    trainer.fit(lightning_module, datamodule=mnist_datamodule)