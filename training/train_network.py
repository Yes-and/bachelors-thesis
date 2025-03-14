from src.actor_critic_model import ActorCritic

import torch
import torch.nn as nn
import torch.optim as optim



def train_policy(g, states, actions, log_probs, rewards):
    """Trains the PPO policy network using the collected experience buffer."""

    # Initialize the policy network and load weights from file
    policy_net = ActorCritic(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE))

    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=g.LR)

    # Convert experience buffer to tensors
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in list(states)]).to(g.DEVICE)
    actions = torch.tensor(list(actions), dtype=torch.long, device=g.DEVICE)
    old_log_probs = torch.stack([torch.tensor(lp, dtype=torch.float32) for lp in list(log_probs)]).to(g.DEVICE)
    returns = torch.tensor(list(rewards), dtype=torch.float32, device=g.DEVICE)

    # Calculate values and advantages
    values = policy_net(states)[1].squeeze()
    advantages = returns - values.detach()

    for _ in range(g.EPOCHS):
        action_probs, values = policy_net(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - g.EPSILON_CLIP, 1 + g.EPSILON_CLIP)
        actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), returns)

        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the updated policy network
    torch.save(policy_net.state_dict(), g.MODEL_PATH)

    return loss.item()
