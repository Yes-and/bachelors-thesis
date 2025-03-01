import datetime
import logging

from pyminion.bots.examples import BigMoney

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from src.bots import MyBot
from src.dqn_model import DQN
from src.game import CustomGame
from src.sets import custom_set
from src.tools import (
    save_replay_buffer_torch, 
    load_replay_buffer_torch,
    sample_experience_batch
)



# Hyperparameters
GAMMA = 0.99 # Discount factor proposed by ChatGPT because Dominion has long-term rewards
LR = 0.001 # Learning rate proposed by ChatGPT

INPUT_SHAPE = 38
N_ACTIONS = 18

BATCH_SIZE = 64 # 32 - 128 Proposed by ChatGPT
MEMORY_SIZE = 20000 # Training should stop at that point. Improvements should appear after 20k, ideal 100k

EPSILON_START = 1 # Full exploration in the beginning
EPSILON_END = 0.05
EPSILON_DECAY = 0.002

LOG_FREQUENCY = 50
TARGET_UPDATE_FREQUENCY = 100 # Update target network every n games
SAVE_BUFFER_FREQUENCY = 5000
SAVE_MODEL_FREQUENCY = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize DQN networks and everything else needed to run successfully
policy_net = DQN(
    input_dim=INPUT_SHAPE,
    output_dim=N_ACTIONS
)
target_net = DQN(
    input_dim=INPUT_SHAPE,
    output_dim=N_ACTIONS
)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target net is only updated periodically
for param in target_net.parameters():
    param.requires_grad = False

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

exp_buffer = load_replay_buffer_torch(
    filename='./replay_buffers/random_replay_buffer.pth'
)

writer = SummaryWriter() # Tensorboard logging



# Function for training the DQN
def train_dqn():
    # Sample a mini-batch from experience replay
    batch = sample_experience_batch(exp_buffer, BATCH_SIZE)
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
    writer.add_scalar("loss_value", loss, game_counter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Function for setting up logging
def initialize_logger(timestamp):
    logger = logging.getLogger('specific_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./logs/run-{timestamp}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger



# Before starting training
timestamp = str(datetime.datetime.now())[:19].replace(":", "-")
logger = initialize_logger(timestamp)
epsilon = EPSILON_START
game_counter = 1
player_VPs = []
enemy_VPs = []



logger.info('Training DQN while playing against BigMoney')
# Main training loop
while True:
    # Logging and saving
    if game_counter>0:
        writer.add_scalar("epsilon", epsilon, game_counter)
        writer.add_scalar("buffer_size", len(exp_buffer), game_counter)
    if game_counter % LOG_FREQUENCY == 0:
        logger.info(f"{game_counter} games have been played so far.")
    if game_counter % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if game_counter % SAVE_BUFFER_FREQUENCY == 0:
        save_replay_buffer_torch(
            exp_buffer, 
            f"./replay_buffers/buffer-{timestamp}.pth"
        )
    if game_counter % SAVE_MODEL_FREQUENCY == 0:
        torch.save(
            policy_net.state_dict(),
            f"./models/model-{timestamp}.pth"
        )

    # Playing a game
    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net, epsilon=epsilon)
    game = CustomGame(
        players=[bot1, bot2],
        expansions=[custom_set],
        exp_buffer=exp_buffer
    )
    result = game.play()
    game_counter += 1

    if game_counter>0:
        writer.add_scalar("player_victory_points", game.player_VPs, game_counter)
        writer.add_scalar("enemy_victory_points", game.enemy_VPs, game_counter)

    # Training
    train_dqn() # train DQN at the end of each game
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY * game_counter)

    if len(exp_buffer)>= MEMORY_SIZE:
        break

writer.close()
torch.save(
    policy_net.state_dict(),
    f"./models/final-model-{timestamp}.pth"    
)
save_replay_buffer_torch(
    exp_buffer,
    f"./replay_buffers/final-buffer-{timestamp}.pth"
)