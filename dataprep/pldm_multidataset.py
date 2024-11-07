import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from typing import List, Dict, Optional, Union, Tuple
import torch

DATASET_ROOT = '/mounted_data/downloaded'

class MultiDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, dataset_idx: int):
        self.dataset = dataset
        self.dataset_idx = dataset_idx
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label, self.dataset_idx, idx
    
    def __len__(self):
        return len(self.dataset)

class MultiDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_configs: List[Dict[str, Union[str, Dict]]],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Args:
            dataset_configs: List of dictionaries containing dataset configurations
                Each dict should have:
                - 'name': Dataset class name (e.g., 'CIFAR10', 'FashionMNIST')
                - 'transform': Optional transform configuration
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory for dataloaders
        """
        super().__init__()
        self.dataset_configs = dataset_configs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_classes = {
            'CIFAR10': datasets.CIFAR10,
            'FashionMNIST': datasets.FashionMNIST,
            # Add more dataset classes as needed
        }
        
    def _get_transform(self, transform_config: Optional[Dict] = None):
        if transform_config is None:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        # Add custom transform configuration handling here
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**transform_config.get('normalize', {'mean': [0.5], 'std': [0.5]}))
        ])
    
    def _load_dataset(self, split: str) -> Tuple[ConcatDataset, Dict]:
        datasets_list = []
        label_maps = {}
        cumulative_length = 0
        
        for idx, config in enumerate(self.dataset_configs):
            dataset_name = config['name']
            transform = self._get_transform(config.get('transform'))
            
            if dataset_name not in self.dataset_classes:
                raise ValueError(f"Dataset {dataset_name} not supported")
            
            dataset = self.dataset_classes[dataset_name](
                root='data',
                train=(split == 'train'),
                download=True,
                transform=transform
            )
            
            wrapped_dataset = MultiDatasetWrapper(dataset, idx)
            datasets_list.append(wrapped_dataset)
            
            # Store label mapping information
            label_maps[idx] = {
                'name': dataset_name,
                'num_classes': len(dataset.classes),
                'classes': dataset.classes,
                'offset': cumulative_length
            }
            cumulative_length += len(dataset)
        
        return ConcatDataset(datasets_list), label_maps

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.train_label_maps = self._load_dataset('train')
            self.val_dataset, self.val_label_maps = self._load_dataset('val')
        
        if stage == 'test' or stage is None:
            self.test_dataset, self.test_label_maps = self._load_dataset('test')
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_label_maps(self):
        """Returns the label mappings for each dataset"""
        return {
            'train': self.train_label_maps,
            'val': self.val_label_maps,
            'test': self.test_label_maps
        }
    

if __name__ == '__main__':
    # Example configuration
    dataset_configs = [
        {
            'name': 'CIFAR10',
            'transform': {
                'normalize': {
                    'mean': [0.4914, 0.4822, 0.4465],
                    'std': [0.2023, 0.1994, 0.2010]
                }
            }
        },
        {
            'name': 'FashionMNIST',
            'transform': {
                'normalize': {
                    'mean': [0.2860],
                    'std': [0.3530]
                }
            }
        }
    ]

    # Create the datamodule
    datamodule = MultiDatasetModule(dataset_configs)

    # Access the data
    for batch in datamodule.train_dataloader():
        images, labels, dataset_indices, sample_indices = batch
        # images: the actual data
        # labels: original labels from each dataset
        # dataset_indices: which dataset each sample came from
        # sample_indices: original indices in their respective datasets