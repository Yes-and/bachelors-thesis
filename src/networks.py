import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



# DQN network
class DQN(nn.Module):
    """
    This is a class for the deep Q-Network used by the DQN RL agent.

    The "get_action" method is called from "my_bot.py" for selecting an action.
    It requires the epsilon parameter for making random choices, which is passed
    together with global variables. Also, an action mask is passed to prevent
    illegal choices.
    """
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Q-value head
        self.q_values = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        """
        Returns Q-values for all actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.q_values(x)
        return q_values

    def get_action(self, state, g, mask=None):
        """
        This network uses epsilon-greedy action selection.

        The input parameters contain the state as a torch vector,
        global variables g with EPSILON and a mask for valid actions.
        """
        if torch.rand(1).item() < g.EPSILON:
            if mask is not None:
                valid_actions = torch.nonzero(mask).squeeze(-1)
                action = valid_actions[torch.randint(0, len(valid_actions), (1,))][0, -1].item()
            else:
                action = torch.randint(0, self.q_values.out_features, (1,)).item()
            return action, {"random": True}

        with torch.no_grad():
            q_values = self.forward(state)
            if mask is not None:
                invalid_mask = (mask == 0)
                q_values = q_values.masked_fill(invalid_mask, -1e9)

            action = torch.argmax(q_values).item()

        return action, {"random": False, "q_values": q_values.detach()}

# Actor-Critic network with shared body
class A2C(nn.Module):
    """
    This is the class for the actor-critic RL architecture,
    which has a shared body. It has separate heads for the actor
    (also called the policy network) and the critic (also called
    the value network.)

    Function "get_action" is called from "my_bot.py" to obtain
    a decision for the buying phase.
    """
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(A2C, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_size)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Returns both action logits and state value.
        """
        x = F.relu(self.fc1(state))
        
        logits = self.actor(x) # For action selection
        value = self.critic(x).squeeze(-1)  # For state value
        
        return logits, value

    def get_action(self, state, g, mask=None):
        """
        A function used for making decisions for the RL agent.

        It has an input variable state for a tensor of the state 
        of the environment and for global variables g which are
        currently not used.
        """
        logits, value = self.forward(state)

        if mask is not None:
            invalid_mask = (mask == 0)
            logits = logits.masked_fill(invalid_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), {"log_prob": log_prob, "probs": probs.detach(), "value": value}

# REINFORCE network
class REINFORCE(nn.Module):
    """
    This is the class for the agent using REINFORCE.

    It contains a policy network and a function for
    obtaining the action for the agent which is called
    from "my_bot.py"
    """
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)
        # self.fc3 = nn.Linear(hidden_dim, action_size)

        # Bias toward 'do_nothing'
        # with torch.no_grad():
        #     self.fc2.bias.fill_(-5.0)   # Low score for all actions
        #     self.fc2.bias[-1] = 5.0      # High score for 'do_nothing'

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        logits = self.fc2(x)
        return logits

    def get_action(self, state, g, mask=None):
        """
        This is the function for obtaining action decisions
        for the REINFORCE agent.

        The input parameters contain a vector of state of the
        environment and global variables g which are unused.
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

        return action.item(), {"log_prob": log_prob, "probs": probs.detach()}
