#%%
from typing import List, Tuple

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        in_channels: int = 1,
        hidden_dims: List[int] = None,
        **kwargs
    ) -> None:
        super(LinearVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.in_channels = in_channels

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1] * in_channels, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * in_channels, latent_dim)

        # Decoder
        decoder_layers = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * in_channels)

        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i - 1]),
                    nn.BatchNorm1d(hidden_dims[i - 1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*decoder_layers)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, in_channels, input_dim)
        batch_size, _, _ = x.shape
        x = x.view(batch_size * self.in_channels, -1)  # Flatten channels into batch dimension
        result = self.encoder(x)
        result = result.view(batch_size, -1)  # Combine results from all channels
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1])  # Reshape for channel-wise processing
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view(-1, self.in_channels, self.input_dim)  # Reshape to (batch_size, in_channels, input_dim)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), x, mu, log_var

    def loss_function(self, recons, x, mu, log_var, kld_weight=0.005, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, device: str, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]

#%%
if __name__ == "__main__":
    # Example setup for LinearVAE
    vae_config = {
        'input_dim': 784,  # 28x28 flattened
        'latent_dim': 2,
        'in_channels': 1,
        'hidden_dims': [512, 256, 128, 64, 32, 16]
    }

    vae_model = LinearVAE(**vae_config)

    # Create a dummy input (batch size 4, 1 channels, 784 flattened input)
    dummy_input = torch.randn(4, *(vae_config[k] for k in ['in_channels', 'input_dim']))

    # Perform a forward pass
    output = vae_model(dummy_input)
    print("output[0].shape =", output[0].shape)  # Reconstructed output
    print("output[1].shape =", output[1].shape)  # Original input
    print("output[2].shape =", output[2].shape)  # Mu
    print("output[3].shape =", output[3].shape)  # Log Var
# %%
