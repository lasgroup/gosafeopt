import torch
import pandas as pd
import gosafeopt


class Data:
    """
    This class should hold all the data needed for training as the optimizers and aquisition functions should not store any data.
    """

    def __init__(
        self,
        train_x=None,
        train_y=None,
        train_x_rollout=None,
        train_y_rollout=None,
        failed_k=None,
        k_rollout=None,
        failed_x_rollout=None,
    ):
        self.train_x = train_x
        self.train_y = train_y

        # TODO could be done more memory efficient
        self.backup = train_x_rollout.to(gosafeopt.device) if train_x_rollout is not None else None
        self.backup_loss = train_y_rollout.to(gosafeopt.device) if train_y_rollout is not None else None
        self.backup_k = k_rollout.to(gosafeopt.device) if k_rollout is not None else None
        self.failed_k = failed_k
        self.failed_x_rollout = failed_x_rollout

    def append_data(self, train_x, train_y):
        if self.train_x is None:
            self.train_x = train_x
            self.train_y = train_y
        else:
            self.train_x = torch.vstack([self.train_x, train_x])
            self.train_y = torch.vstack([self.train_y, train_y])

    def append_failed(self, failed_k, failed_x_rollout):
        if self.failed_k is None:
            self.failed_k = failed_k.reshape(1, -1)
            self.failed_x_rollout = failed_x_rollout.reshape(1, -1)
        else:
            self.failed_k = torch.vstack([self.failed_k, failed_k])
            self.failed_x_rollout = torch.vstack([self.failed_x_rollout, failed_x_rollout])

    def append_backup(self, train_x_rollout, train_y_rollout, k_rollout):
        if self.backup is None:
            self.backup = train_x_rollout
            self.backup_loss = train_y_rollout.repeat(train_x_rollout.shape[0], 1)
            self.backup_k = k_rollout.repeat(train_x_rollout.shape[0], 1)
        else:
            self.backup = torch.vstack([self.backup, train_x_rollout])
            self.backup_loss = torch.vstack([self.backup_loss, train_y_rollout.repeat(train_x_rollout.shape[0], 1)])
            self.backup_k = torch.vstack([self.backup_k, k_rollout.repeat(train_x_rollout.shape[0], 1)])

    def resetForNewGate(self):
        self.backup = None
        self.backup_loss = None
        self.backup_k = None
        self.failed_k = None
        self.failed_x_rollout = None

    def save(self, folder=""):
        features = pd.DataFrame(self.train_x.to("cpu"))
        labels = pd.DataFrame(self.train_y.to("cpu"))
        features.to_csv(f"{folder}/train_x.csv", index=False)
        labels.to_csv(f"{folder}/train_y.csv", index=False)

        if self.failed_k is not None:
            features = pd.DataFrame(self.failed_k.to("cpu"))
            features.to_csv(f"{folder}/failed.csv", index=False)

    def load(self, folder=""):
        features = pd.read_csv(f"{folder}/train_x.csv")
        labels = pd.read_csv(f"{folder}/train_y.csv")

        self.train_x = torch.from_numpy(features.to_numpy())
        self.train_y = torch.from_numpy(labels.to_numpy())
