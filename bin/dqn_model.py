import torch
import torch.nn as nn
import numpy as np



# Made by ChatGPT
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        input_size = int(np.prod(input_shape))  # Flatten the input shape into a single dimension

        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),  # First hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),  # Second hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)  # Output layer
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)  # Flatten the input
        return self.fc(x)