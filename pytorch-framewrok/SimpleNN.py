import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as f
from torch import optim
import torchvision


class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(128 * 128,  64 * 64)
        self.fc2 = nn.Linear(64 * 64, 32 * 32)
        self.fc3 = nn.Linear(32 * 32,  16 * 16)
        self.fc4 = nn.Linear(16 * 16, 120)

    def forward(self, x):
        # h1 = relu(xw +b)
        x = f.relu(self.fc1(x))
        # h2 = relu(h1w2 +b2)
        x = f.relu(self.fc2(x))
        # h3 = h2w3 + b3
        x = f.relu((self.fc3(x)))
        # # h4 = h3w4 + b4
        # x = f.relu((self.fc4(x)))

        x = self.fc4(x)

        return x
