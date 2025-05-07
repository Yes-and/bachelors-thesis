import numpy as np
import torch
import torch.nn.functional as F



def DQN_train(g, q_net, optimizer, states, actions, rewards, next_states, dones, target_net=None):
    """
    This function is used for training the network for the DQN agent.

    The input parameters are as follows:
    * g = global variables
    * q_net = the DQN
    * optimizer = initialized optimizer for the DQN
    * ...
    * the target_net may be used in case of Dueling DQN architecture
    """

    # Convert to tensors
    states = torch.as_tensor(states, dtype=torch.float32, device=g.DEVICE)
    next_states = torch.as_tensor(next_states, dtype=torch.float32, device=g.DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.long, device=g.DEVICE).unsqueeze(-1)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=g.DEVICE).unsqueeze(-1)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=g.DEVICE).unsqueeze(-1)

    # Compute predicted Q-values for current states and actions
    q_values = q_net(states)
    q_value = q_values.gather(1, actions)

    # Compute target Q-values
    with torch.no_grad():
        if target_net is not None:
            next_q_values = target_net(next_states)
        else:
            next_q_values = q_net(next_states)

        # Get max Q-value for the next state
        max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]

        # Bellman target
        target_q_value = rewards + (1 - dones) * g.GAMMA * max_next_q_values

    # Loss: mean squared TD error
    loss = F.mse_loss(q_value, target_q_value)

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()

    # Save the updated Q-network
    torch.save(q_net.state_dict(), g.MODEL_PATH)

    return loss.item(), np.array(q_values.detach()).mean(), np.array(q_values.detach()).std()

def A2C_train(g, shared_net, optimizer, states, actions, rewards, next_states, dones, value_coef=0.5, entropy_coef=0.01):
    """
    This function is used by the A2C agent to train the complex network.

    The input parameters are as follows:
    * g = global variables
    * shared_net = complex network used by the A2C
    * optimizer = initialized optimizer used by the A2C
    * ...
    * value_coef = hyperparameter which adjusts the training
    speed of the actor
    * entropy_coef = hyperparameter which adjusts the training
    speed of the critic 
    """

    states = torch.as_tensor(states, dtype=torch.float32, device=g.DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.long, device=g.DEVICE)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=g.DEVICE)
    next_states = torch.as_tensor(next_states, dtype=torch.float32, device=g.DEVICE)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=g.DEVICE)

    # Normalize rewards if needed
    # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Get logits and state values
    logits, state_values = shared_net(states)

    # Get state values for next states
    with torch.no_grad():
        _, next_state_values = shared_net(next_states)

    # Compute TD target
    targets = rewards + g.GAMMA * next_state_values * (1 - dones)

    # Advantage, also called TD error
    advantages = targets - state_values

    # Get log probs of the actions taken
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # Policy loss (actor)
    policy_loss = -torch.mean(log_probs * advantages.detach())

    # Value loss (critic)
    value_loss = F.mse_loss(state_values, targets)

    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(shared_net.parameters(), max_norm=1.0)

    optimizer.step()

    # Save the updated model
    torch.save(shared_net.state_dict(), g.MODEL_PATH)

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

def REINFORCE_train(g, policy_net, optimizer, states, actions, rewards):
    """
    Function used to train the network used by the REINFORCE agent.

    Input parameters are as follows:
    * g = global variables
    * policy_net = policy network used by REINFORCE
    * optimizer = initialized optimizer used by REINFORCE
    * ...
    """

    states = torch.as_tensor(states, dtype=torch.float32, device=g.DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.long, device=g.DEVICE)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=g.DEVICE)

    # Normalize rewards
    # zero mean, unit variance
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Get logits
    logits = policy_net(states)

    # Get log probs of the actions taken
    probs = F.softmax(logits)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    action_entropy = dist.entropy().mean()

    # Compute loss
    # encourage good actions, suppress bad ones
    loss = -torch.mean(log_probs * rewards)

    # Backpropagate and update
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients to ensure small updates
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    optimizer.step()

    # Save the updated policy network
    torch.save(policy_net.state_dict(), g.MODEL_PATH)

    return loss.item(), action_entropy.item()
