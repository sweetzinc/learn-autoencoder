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

from models.vanilla_vae import VanillaVAE
from lightning_vae import VAELightning
from lightningdata_mnist import MNISTDataModule


if __name__ == '__main__' :

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
    log_config = {'save_dir': '/workspace/torch_vae/logs/', 'name': 'VanillaVAE'}

    vae_model = VanillaVAE(**vae_config)
    lightning_module = VAELightning(vae_model, lightning_config)
    mnist_datamodule = MNISTDataModule(data_dir=data_config['data_dirpath'], batch_size=data_config['batch_size'])

    # Prepare for training
    pl.seed_everything(lightning_config['manual_seed'], True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)


    # Set up Training
    tb_logger = TensorBoardLogger(save_dir=log_config['save_dir'],
                                name=log_config['name'],)

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

    mnist_datamodule.setup('fit')
    mnist_datamodule.setup('test')
    trainer.fit(lightning_module, datamodule=mnist_datamodule)

    # Save the configuration
    config = {
        'vae_config': vae_config,
        'lightning_config': lightning_config,
        'data_config': data_config,
        'log_config': log_config
    }
    with open(os.path.join(log_config['save_dir'], "configs", 'config.yml'), 'w') as config_file:
        yaml.dump(config, config_file)