# from src.actor_critic_model import ActorCritic

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
# import torch.optim as optim



def train_policy(g, policy_net, optimizer, states, actions, log_probs, advantages, state_values):
    """Trains the PPO policy network using the collected experience buffer."""

    # Initialize the policy network and load weights from file
    # policy_net = ActorCritic(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    # policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE))

    # Initialize optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=g.LR)

    # Convert experience buffer to tensors
    states = torch.as_tensor(states, dtype=torch.float32, device=g.DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.long, device=g.DEVICE).unsqueeze(1)
    old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=g.DEVICE).squeeze()

    # Normalize advantages
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=g.DEVICE)
    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    state_values = torch.as_tensor(state_values, dtype=torch.float32, device=g.DEVICE)
    returns = advantages + state_values

    for _ in range(g.EPOCHS):
        # Shuffle experience buffer
        indices = np.arange(len(states))
        np.random.shuffle(indices)

        # Select mini-batches for learning
        for i in range(0, len(states), g.BATCH_SIZE):
            batch_indices = indices[i:i + g.BATCH_SIZE]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = norm_advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Get action probabilities and values
            action_probs, values = policy_net(batch_states)
        
            # Get action entropy for logging
            dist = Categorical(probs=action_probs)
            action_entropy = dist.entropy().mean()

            action_log_probs = torch.log(action_probs.gather(1, batch_actions)).squeeze()
            ratio = torch.exp(action_log_probs - batch_old_log_probs)

            clipped = torch.clamp(ratio, 1 - g.EPSILON_CLIP, 1 + g.EPSILON_CLIP)
            actor_loss = -torch.min(ratio * batch_advantages, clipped * batch_advantages).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), batch_returns.detach())

            loss = actor_loss + 0.5 * critic_loss
            optimizer.zero_grad()
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
            optimizer.step()

    # Save the updated policy network
    torch.save(policy_net.state_dict(), g.MODEL_PATH)

    return loss.item(), actor_loss.item(), critic_loss.item(), action_entropy.item()
