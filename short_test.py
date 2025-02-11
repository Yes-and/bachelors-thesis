import sys
import os
import logging

from pyminion.bots.examples import BigMoney

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np

from src.bots import MyBot
from src.data_structures import ExperienceBuffer
from src.dqn_model import DQN
from src.game import CustomGame
from src.sets import custom_set



# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 1e-3  # Learning rate

INPUT_SHAPE = 38
N_ACTIONS = 18

BATCH_SIZE = 64
MEMORY_SIZE = 5000

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQ = 10
MAX_EPISODES = 1000
MAX_STEPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


policy_net = DQN(
    input_size=INPUT_SHAPE,
    output_size=N_ACTIONS
)
target_net = DQN(
    input_size=INPUT_SHAPE,
    output_size=N_ACTIONS
)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target net is only updated periodically

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

exp_buffer = ExperienceBuffer(capacity=MEMORY_SIZE)

# TensorBoard logging
writer = SummaryWriter()

def train_dqn():
    if len(exp_buffer) < BATCH_SIZE:
        return  # Skip training until we have enough experiences

    # Sample a mini-batch from experience replay
    batch = exp_buffer.sample(BATCH_SIZE)
    states, actions, rewards, dones, next_states = batch[0], batch[1], batch[2], batch[3], batch[4]

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)

    # Compute Q-values for the actions taken: Q(s, a)
    q_values = policy_net(states).gather(1, actions-1)

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))  # Bellman Equation

    # Compute loss and optimize
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon = EPSILON_START
num_episodes = 1000
total_rewards = []
victory_points = []

# Set up logging
logger = logging.getLogger('specific_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('progress_log.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

for episode in range(num_episodes):
    if episode>0:
        logger.info(f"Current episode {episode} of {num_episodes}")
        writer.add_scalar("epsilon", epsilon, episode)
        m_reward = np.mean(total_rewards[-100:])
        writer.add_scalar("reward_100", m_reward, episode)
        l_vp = victory_points[-1]
        writer.add_scalar("victory points", l_vp, episode)

    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net, epsilon=0.05)

    game = CustomGame(
        players=[bot1, bot2],
        expansions=[custom_set],
        exp_buffer=exp_buffer
    )
    result = game.play()
    total_rewards += game.total_rewards
    victory_points.append(game.victory_points)
    train_dqn()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print(f"Episode {episode}: Epsilon = {epsilon:.3f}")

writer.close()

# Save the trained model
torch.save(policy_net.state_dict(), "new_dqn_dominion.pth")
