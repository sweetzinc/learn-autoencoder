#%%
from typing import List, Callable, Union, Any, TypeVar, Tuple

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F

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

#%%
if __name__ == "__main__" : 

    # Example Set up vanilla vae 
    vae_config = {
        'in_channels': 1,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 28, 'height': 28}

    vae_model = VanillaVAE(**vae_config)

    # Create a dummy input (batch size 4, 1 channel, 28x28 image as in MNIST dataset)
    dummy_input = torch.randn(4, *(vae_config[k] for k in['in_channels', 'width', 'height']))

    # Perform a forward pass
    output = vae_model(dummy_input)
    print("output[0].shape=", output[0].shape)
    print("output[1].shape=", output[1].shape)
    print("output[2].shape=", output[2].shape) 
    print("output[3].shape=", output[3].shape)   # Check reconstruction and input shapes
# %%
