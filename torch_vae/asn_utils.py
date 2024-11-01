import os
from pathlib import Path
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

class ConfigSaverCallback(Callback): 
    def __init__(self, config: dict, save_dir: str):
        super().__init__()
        self.config = config
        self.save_dir = save_dir

    def on_fit_start(self, trainer, pl_module):
        config_path = os.path.join(self.save_dir, "configs")
        Path(config_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(config_path, 'config.yml'), 'w') as config_file:
            yaml.dump(self.config, config_file)
        print(f"Configuration saved to {config_path}/config.yml")