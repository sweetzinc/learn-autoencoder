#%%
import os 
from pathlib import Path
from typing import List, Callable, Union, Any, TypeVar, Tuple
import yaml
# PyTorch
import torch
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append('/workspace/torch_vae')
from models import vae_models 
from pl_linvae import LVAELightning 
from lightningdata_mnist import MNISTDataModule
from asn_utils import ConfigSaverCallback


if __name__ == '__main__' :
    print("torch version=", torch.__version__)
    print("torch.cuda.is_available() = ", torch.cuda.is_available())

    vae_config = {
        'name': 'LinearVAE',
        'input_dim': 784,  # 28x28 flattened
        'latent_dim': 2,
        'in_channels': 1,
        'hidden_dims': [512, 256, 128, 64, 32, 16]
    }
    lightning_config = {
        'LR': 0.005,
        'weight_decay': 0.0,
        'scheduler_gamma': 0.95,
        'kld_weight': 0.00025,
        'manual_seed': 1265 }
    
    data_config = { 'data_dir':"/mounted_data/downloaded", 'batch_size':256 , 'flatten': True}
    log_config = {'save_dir': '/workspace/torch_vae/logs/', 'name': 'LinearVAE'}
    config = {
        'vae_config': vae_config,
        'lightning_config': lightning_config,
        'data_config': data_config,
        'log_config': log_config
    }
    vae_model = vae_models[vae_config['name']](**vae_config)
    lightning_module = LVAELightning(vae_model, lightning_config)
    # mnist_datamodule = MNISTDataModule(data_dir=data_config['data_dirpath'], batch_size=data_config['batch_size'])
    mnist_datamodule = MNISTDataModule(**data_config)

    # Prepare for training
    pl.seed_everything(lightning_config['manual_seed'], True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)


    # Set up Training
    tb_logger = TensorBoardLogger(**log_config)
    # Initialize Callback
    config_saver = ConfigSaverCallback(config=config, save_dir=tb_logger.log_dir)

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor("epoch"),
            ModelCheckpoint(save_top_k=2,
                            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                            monitor="val_loss",
                            save_last=True),
            config_saver,  # Add the custom callback here
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=100,
        check_val_every_n_epoch=1)

    mnist_datamodule.setup('fit')
    # mnist_datamodule.setup('test')
    trainer.fit(lightning_module, datamodule=mnist_datamodule)

