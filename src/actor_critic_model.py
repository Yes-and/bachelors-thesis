import torch
import torch.nn as nn



# Actor-Critic (Shared Network)
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head (outputs action probabilities)
        self.actor = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=-1)

        # Critic head (outputs state value estimate)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.actor(x)), self.critic(x)  # (policy, value estimate)