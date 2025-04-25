import torch
import torch.nn.functional as F



def train_policy(g, policy_net, optimizer, states, actions, turns, end_turns, end_rewards):
    """
    policy_net: instance of PolicyNetwork
    optimizer: PyTorch optimizer
    batch: list of 64 tuples (state, action, reward)
           where:
               - state is a tensor of shape [state_dim]
               - action is an int
               - reward is a float (final game outcome)
    """

    states = torch.as_tensor(states, dtype=torch.float32, device=g.DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.long, device=g.DEVICE)
    rewards = torch.as_tensor(end_rewards, dtype=torch.float32, device=g.DEVICE)

    # Normalize rewards: zero mean, unit variance
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Forward pass: get logits
    logits = policy_net(states)

    # Get log probs of the actions actually taken
    probs = F.softmax(logits)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    action_entropy = dist.entropy().mean()

    # Compute loss: encourage good actions, suppress bad ones
    loss = -torch.mean(log_probs * rewards)

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()

    # Optional: clip gradients to ensure small updates
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    optimizer.step()

    # Save the updated policy network
    torch.save(policy_net.state_dict(), g.MODEL_PATH)

    return loss.item(), action_entropy.item()
