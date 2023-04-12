import torch
from torch import nn


class MLP_Nerf(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(64, 17)

    def forward(self, point):
        x = self.fc1(point)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return self.output_layer(x)
