from typing import Optional

import numpy as np
import wandb
from torch import Tensor

from gosafeopt.tools.data import Data


class WandbLogger:
    def __init__(self, config: dict, project: str, directory: str, save_interval: int) -> None:
        self.save_interval = save_interval
        wandb.init(project=project, config=config, dir=directory)
        wandb.save()

    def log(
        self,
        x: Tensor,
        y: Tensor,
        reward_max: Tensor,
        loss_aq: Tensor,
        backup_triggered: bool,
        data: Data,
        additional: Optional[dict] = None,
        episode: Optional[int] = None,
    ) -> None:
        if wandb.run is None:
            raise Exception("Wandb has no run yet.")

        log_data = {
            "loss_aq": loss_aq,
            "backup_triggered": int(backup_triggered),
            "k": x,
            "reward": y,
            "logreward": np.log(np.abs(y[0])),
            "rewardMax": reward_max[0],
            "logRewardMax": np.log(np.abs(reward_max[0])),
        }

        if additional is not None:
            for key, value in additional.items():
                log_data[key] = value

        if len(y) > 1:
            for i in range(1, len(y)):
                log_data[f"Constraint {i}"] = y[i]

        # Save progress
        if episode is not None and episode > 0 and not episode % self.save_interval:
            data.save(f"{wandb.run.dir}")

        wandb.log(data=log_data, step=episode)

    def finish(self):
        wandb.finish()
