#%%
import torch
import torch.nn as nn
from encoders import LinearEncoder

class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim=None):
        """latent_dim: Size of the latent space, input to the decoder."""
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(latent_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU())
            latent_dim = h_dim
        if output_dim is not None:
            # Final layer maps back to original input dimension
            layers.append(nn.Linear(latent_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = LinearEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = LinearDecoder(latent_dim, hidden_dims[::-1], input_dim)

    def forward(self, x, return_latent=False):
        latent = self.encoder(x)  # Latent representation
        if return_latent:
            return latent
        reconstructed = self.decoder(latent)
        return reconstructed

class MultiChannelAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, channels=3):
        """
        Autoencoder handling multiple channels with separate encoders and decoders.

        Args:
            input_dim (int): Number of input features per channel.
            hidden_dims (list of int): Number of units in each encoder layer.
            latent_dim (int): Size of the latent space.
            channels (int): Number of channels (e.g., 3 for RGB).
        """
        super().__init__()
        self.channels = channels

        # Create separate encoders and decoders for each channel
        self.encoders = nn.ModuleList([
            LinearEncoder(input_dim, hidden_dims, latent_dim) for _ in range(channels)
        ])
        self.decoders = nn.ModuleList([
            LinearDecoder(latent_dim, hidden_dims[::-1], input_dim) for _ in range(channels)
        ])

    def forward(self, x, return_latent=False):
        """
        Forward pass of the multi-channel autoencoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, features, channels).
            return_latent (bool): If True, return latent representations instead of reconstructed output.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Reconstructed input or latent representations.
        """
        batch_size, features, channels = x.size()
        assert channels == self.channels, f"Expected {self.channels} channels, got {channels}"

        # Initialize lists to store latent representations and reconstructions
        latent_reps = []
        reconstructions = []

        # Iterate over each channel
        for c in range(channels):
            # Extract data for channel c: shape (batch_size, features)
            channel_data = x[:, :, c]

            # Encode
            latent = self.encoders[c](channel_data)
            latent_reps.append(latent)

            if not return_latent:
                # Decode
                reconstructed = self.decoders[c](latent)
                reconstructions.append(reconstructed)

        if return_latent:
            # Stack latent representations: list of (batch_size, latent_dim) -> (batch_size, latent_dim, channels)
            latent_stack = torch.stack(latent_reps, dim=2)
            return latent_stack  # Shape: (batch_size, latent_dim, channels)
        else:
            # Stack reconstructions: list of (batch_size, features) -> (batch_size, features, channels)
            reconstructed_stack = torch.stack(reconstructions, dim=2)
            return reconstructed_stack  # Shape: (batch_size, features, channels)
#%%
# if __name__ == "__main__":
#     # Configuration
#     input_dim = 100
#     encoder_dims = [64, 32, 16]
#     latent_dim = 8

#     # Model
#     model = Autoencoder(input_dim, encoder_dims, latent_dim)

#     # Dummy data
#     x = torch.randn(10, input_dim)  # Batch size of 10

#     # Get the latent representation
#     latent = model(x, return_latent=True)
#     print(f"Latent shape: {latent.shape}")
#     # Get the reconstructed input
#     reconstructed = model(x, return_latent=False)
#     print(f"Reconstructed shape: {reconstructed.shape}")
# %%
if __name__ == "__main__":
    # Configuration
    input_dim = 100          # Number of input features per channel
    encoder_dims = [64, 32, 16]  # Encoder layer sizes
    latent_dim = 8           # Latent space size
    channels = 3             # Number of channels (e.g., RGB)

    # Instantiate the model
    model = MultiChannelAutoencoder(input_dim=input_dim, hidden_dims=encoder_dims, latent_dim=latent_dim, channels=channels)
    print(model)

    # Test with different batch sizes
    for batch_size in [10, 32]:
        # Dummy data: shape (batch_size, features, channels)
        x = torch.randn(batch_size, input_dim, channels)

        # Get latent representations
        latent = model(x, return_latent=True)
        print(f"Batch size: {batch_size}, Latent shape: {latent.shape}")  # Expected: (batch_size, latent_dim, channels)

        # Get reconstructed input
        reconstructed = model(x, return_latent=False)
        print(f"Batch size: {batch_size}, Reconstructed shape: {reconstructed.shape}")  # Expected: (batch_size, features, channels)
# %%
