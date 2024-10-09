import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path

ROOT_PATH = Path(__file__).parent

WANDB_PATH = ROOT_PATH / "wandb_log"
SUMMARY_PATH = ROOT_PATH / "summary_log"
if not os.path.exists(WANDB_PATH):
    os.mkdir(WANDB_PATH)
if not os.path.exists(SUMMARY_PATH):
    os.mkdir(SUMMARY_PATH)
log_dir = SUMMARY_PATH


class MySummary:

    def __init__(self, log_dir_name="default", use_wandb=True):
        log_path = str(log_dir / log_dir_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.login(key="613f55cae781fb261b18bad5ec25aa65766e6bc8")
            self.wandb_logger = wandb.init(project="RLHF", dir=WANDB_PATH)

    def add_float(self, y, title):
        if self.use_wandb:
            self.wandb_logger.log({title: y})

    def close(self):
        self.writer.close()
