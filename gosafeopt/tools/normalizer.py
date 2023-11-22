import torch


class Normalizer:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit_transform(self, data):
        if self.normalize:
            if data.shape[0] > 1:
                self.mean = data.mean(dim=0, keepdim=True)
                self.std = data.std(dim=0, keepdim=True)
            else:
                self.mean = data.mean(dim=0, keepdim=True)
                self.std = 1
        else:
            self.mean = torch.zeros(1) if data.dim() == 1 else torch.zeros(1, data.shape[1])
            self.std = torch.ones_like(self.mean)
        return self.transform(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def itransform(self, data):
        return data * self.std + self.mean
