import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, existing, missing, gt_shape, epoch, device, noise=None):
        raise NotImplementedError
