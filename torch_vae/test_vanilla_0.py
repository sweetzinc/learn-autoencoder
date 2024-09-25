#%%
import os
import torch
import yaml
from pytorch_lightning import Trainer
from matplotlib import pyplot as plt 
from torchvision import transforms
from torchvision.utils import save_image

from models.vanilla_vae import VanillaVAE
from lightning_vae import VAELightning
from lightningdata_mnist import MNISTDataModule
from torch_vae.lightningdata_engchar import HandwrittenCharDataModule, load_specific_labels

#%%
if __name__ == '__name__' :
    vae_config = {
        'in_channels': 1,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 28, 'height': 28}
    lightning_config = {
        'LR': 0.005,
        'weight_decay': 0.0,
        'scheduler_gamma': 0.95,
        'kld_weight': 0.00025,
        'manual_seed': 1265 }
    data_config = { 'data_dirpath':"/mounted_data/downloaded", 'batch_size':256 }

    # Path to your checkpoint file
    checkpoint_path = '/workspace/torch_vae/logs/VanillaVAE/version_15/checkpoints/last.ckpt'

    # Load the trained LightningModule from the checkpoint
    vae_model = VanillaVAE(**vae_config)
    trained_model = VAELightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        vae_model=vae_model,
        params=lightning_config
    )

    # Set the model to evaluation mode and move to the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model.eval()
    trained_model.to(device)
#%%
    # Get the test dataloader
    dataset_path = "/mounted_data/downloaded"
    mnist_datamodule = MNISTDataModule(data_dir=dataset_path)
    mnist_datamodule.setup('test')
    test_dataloader = mnist_datamodule.test_dataloader()

    # Collect latent variables and labels
    latents = []
    labels_list = []

    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            # Pass through the encoder to get mu and log_var
            mu, log_var = trained_model.model.encode(images)
            # Optionally, sample z using reparameterization
            z = trained_model.model.reparameterize(mu, log_var)
            # Collect z and labels
            latents.append(z.cpu())
            labels_list.append(labels.cpu())

    # Concatenate all latents and labels
    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Convert to NumPy arrays for plotting
    latents_np = latents.numpy()
    labels_np = labels.numpy()

    # Plot the latent space
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latents_np[:, 0], latents_np[:, 1], c=labels_np, cmap='tab10', s=5)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.show()

#%%
if True:
    engchar_datamodule = HandwrittenCharDataModule(dataset_dir='/mounted_data/downloaded/EngLetters',
                                        batch_size=256)
    images, labels = load_specific_labels(engchar_datamodule, ['K'])
    images = torch.cat(images, dim=0)
    images = images[:,None,:,:]
    latents = []
    with torch.no_grad():
        images = images.to(device)
        # Pass through the encoder to get mu and log_var
        mu, log_var = trained_model.model.encode(images)
        # Optionally, sample z using reparameterization
        z = trained_model.model.reparameterize(mu, log_var)
        # Collect z and labels
        latents.append(z.cpu())

    # Concatenate all latents and labels
    latents = torch.cat(latents, dim=0)

    # Convert to NumPy arrays for plotting
    char_latents_np = latents.numpy()

    # Plot the latent space
    plt.figure(figsize=(10, 10))
    scatter1 = plt.scatter(latents_np[:, 0], latents_np[:, 1], c=labels_np, cmap='tab10', s=5, label='MNIST Digits')
    scatter2 = plt.scatter(char_latents_np[:, 0], char_latents_np[:, 1], marker='x', s=50, color='red', label='EngChar K')
    plt.legend()

    # plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')



# %%
# Show first 5 images and labels
images = images.cpu().numpy()
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    ax = axes[i]
    img = images[i].squeeze(0)  # Remove the channel dimension
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {labels[i]}')
    ax.axis('off')
plt.show()
# %%

