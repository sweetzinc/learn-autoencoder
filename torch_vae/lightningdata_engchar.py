#%%
import os
import pandas as pd
import json 
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
import matplotlib.pyplot as plt 

class HandwrittenCharDataset(Dataset):
    def __init__(self, dataset_dir, csv_file='labels.csv', transform=None):
        self.image_dir = os.path.join(dataset_dir, 'crop_rescale')
        self.data = pd.read_csv(os.path.join(dataset_dir, csv_file), header=0)
        self.transform = transform
        mapping_fpath = os.path.join(dataset_dir, 'mapping.json')
        with open(mapping_fpath, 'r') as f:
            self.mapping = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HandwrittenCharDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((189.3025,), (103.9265,))
        ])

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = HandwrittenCharDataset(
                dataset_dir=self.dataset_dir,
                csv_file='labels.csv',
                transform=self.transform
            )
            
            # You might want to split this into train and validation sets
            self.val_dataset = self.train_dataset
        
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = HandwrittenCharDataset(
                dataset_dir=self.dataset_dir,
                csv_file='labels.csv',
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def show_images(datamodule, num_images=25):
    # Setup the datamodule
    datamodule.setup()
    
    # Get a batch of images
    dataloader = datamodule.train_dataloader()
    images, labels = next(iter(dataloader))
    
    # Select a subset of images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Convert labels to strings (assuming they're integers)
    label_strings = [str(label) for label in labels]
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images, nrow=5, normalize=True, padding=2)
    
    # Convert the grid to numpy and transpose it for display
    grid_np = grid.numpy().transpose((1, 2, 0))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid_np)
    
    # Remove axes
    ax.axis('off')
    
    # Add labels to each image
    for i, label in enumerate(label_strings):
        row = i // 5
        col = i % 5
        ax.text(col * (images.size(2) + 2) + images.size(2)/2, 
                row * (images.size(3) + 2) + images.size(3) + 2, 
                label, color='white', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.7, pad=0.5))
    
    plt.tight_layout()
    plt.show()

def load_specific_labels(datamodule, target_labels):
    # Setup the datamodule
    datamodule.setup()
    
    # Get a batch of images
    dataloader = datamodule.train_dataloader()
    
    specific_images = []
    specific_labels = []
    
    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            if lbl in target_labels:
                specific_images.append(img)
                specific_labels.append(lbl)
                
    return specific_images, specific_labels

    # Example usage
    if __name__ == "__main__":
        data_dir = '/mounted_data/downloaded/EngLetters'
        datamodule = HandwrittenCharDataModule(dataset_dir=data_dir, batch_size=25)
        target_labels = [0, 1, 2]  # Example target labels
        images, labels = load_specific_labels(datamodule, target_labels)
        
        # Display the images
        num_images_to_display = len(images)
        fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 15))
        for i, img in enumerate(images):
            axes[i].imshow(img.permute(1, 2, 0).numpy(), cmap='gray')
            axes[i].axis('off')
        plt.show()
#%% Test Dataset   
if __name__ == "__main__" : 
    data_dir = '/mounted_data/downloaded/EngLetters'
    dataset = HandwrittenCharDataset(dataset_dir=data_dir, csv_file='labels.csv', transform=None)

    # Load a few images
    num_images_to_display = 5
    images = [dataset[600+i][0] for i in range(num_images_to_display)]

    # Display the images
    fig, axes = plt.subplots(1, num_images_to_display)
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.show()

#%% Test Datamodule   
if 0 :#__name__ == "__main__" : 
    data_dir = '/mounted_data/downloaded/EngLetters'
    datamodule = HandwrittenCharDataModule(dataset_dir=data_dir, batch_size=25)
    show_images(datamodule)

# %%
