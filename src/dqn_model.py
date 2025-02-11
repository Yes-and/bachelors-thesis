import torch.nn as nn
import torch.nn.functional as F



# Proposed by ChatGPT
class DQN(nn.Module):
    def __init__(self, input_size=36, output_size=17, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output Q-values for actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Raw Q-values for each action
