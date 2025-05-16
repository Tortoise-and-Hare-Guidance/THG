from torch import nn


class DiffusionModel(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
