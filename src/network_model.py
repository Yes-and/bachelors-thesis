import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Policy Network (Actor Only)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)
        # self.fc3 = nn.Linear(hidden_dim, action_size)

        # Bias toward 'do_nothing' (assume index 0)
        # with torch.no_grad():
        #     self.fc2.bias.fill_(-5.0)   # Low score for all actions
        #     self.fc2.bias[-1] = 5.0      # High score for 'do_nothing'

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        logits = self.fc2(x)  # Shape: [batch_size, num_actions]
        return logits

    def get_action(self, state, mask=None):
        """
        state: tensor of shape [state_dim]
        mask: tensor of shape [num_actions], with 1 for valid, 0 for invalid
        """
        logits = self.forward(state)

        if mask is not None:
            # Apply a large negative value to masked-out actions
            invalid_mask = (mask == 0)
            logits = logits.masked_fill(invalid_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, probs.detach()
