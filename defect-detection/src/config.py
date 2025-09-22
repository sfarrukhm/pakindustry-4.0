import os
import yaml
import random
import numpy as np
import torch

class Config:
    def __init__(self, config_path="config.yaml", section="default"):
        # Load YAML
        with open(config_path, "r") as f:
            all_cfg = yaml.safe_load(f)
        cfg = all_cfg.get(section, {})

        # Store config dict
        self._cfg = cfg

        # Seed everything
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Resolve paths (absolute for portability)
        self.data_dir = os.path.abspath(cfg["data_dir"])
        self.csv_train = os.path.abspath(cfg["csv_train"])
        self.csv_valid = os.path.abspath(cfg["csv_valid"])
        self.img_dir_train = os.path.abspath(cfg["img_dir_train"])
        self.img_dir_valid = os.path.abspath(cfg["img_dir_valid"])
        self.save_path = os.path.abspath(cfg["save_path"])

        # Training params
        self.batch_size = cfg.get("batch_size", 32)
        self.img_size = cfg.get("img_size", 300)
        self.epochs = cfg.get("epochs", 20)
        self.lr = cfg.get("lr", 0.001)
        self.patience = cfg.get("patience", 3)

    def as_dict(self):
        """Return config as a dictionary (useful for logging)"""
        return self._cfg
