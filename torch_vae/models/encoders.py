import torch
import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim=None):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU())
            input_dim = h_dim
        # Final layer maps to latent dimension
        if latent_dim is not None:
            layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
