import datetime
import logging
import multiprocessing
import concurrent.futures

from pyminion.bots.examples import BigMoney

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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
MEMORY_SIZE = 50000 # Training should stop at that point. Improvements should appear after 20k, ideal 100k

EPSILON_START = 1 # Full exploration in the beginning
EPSILON_END = 0.05
EPSILON_DECAY = 0.002

LOG_FREQUENCY = 100
NUM_GAMES_PARALLEL = 8
TRAIN_FREQUENCY = 10 # Train network every n games
TARGET_UPDATE_FREQUENCY = 200 # Update target network every n games
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

# Initialize a multiprocessing manager
manager = multiprocessing.Manager()
shared_exp_buffer = manager.list() # Thread safe alternative to deque
buffer_lock = multiprocessing.Lock() # Lock for buffer safety

# Load initial replay buffer
shared_exp_buffer.extend(
    load_replay_buffer_torch(
        filename='./replay_buffers/random_replay_buffer.pth'
    )
)

writer = SummaryWriter() # Tensorboard logging



# Function for training the DQN
def train_dqn():
    # Sample a mini-batch from experience replay
    batch = sample_experience_batch(shared_exp_buffer, BATCH_SIZE)
    states, actions, rewards, dones, next_states = batch

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

    logger.info("Fitting the DQN.")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("Fitting has finished.")

# Function for setting up logging
def initialize_logger(timestamp):
    logger = logging.getLogger('specific_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./logs/run-{timestamp}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# Function to run a game in parallel
def play_game(policy_net, epsilon):
    local_exp_buffer = deque()

    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net, epsilon=epsilon)
    game = CustomGame(
        players=[bot1, bot2],
        expansions=[custom_set],
        exp_buffer=local_exp_buffer
    )
    result = game.play() # Fills local_exp_buffer
    return list(local_exp_buffer)



# Before starting training
timestamp = str(datetime.datetime.now())[:19].replace(":", "-")
logger = initialize_logger(timestamp)
epsilon = EPSILON_START
game_counter = 1
player_VPs = []
enemy_VPs = []


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    logger.info('Training DQN while playing against BigMoney')

    # Main training loop
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GAMES_PARALLEL) as executor:
        next_log = LOG_FREQUENCY
        next_train = TRAIN_FREQUENCY
        next_target_update = TARGET_UPDATE_FREQUENCY
        next_save_buffer = SAVE_BUFFER_FREQUENCY
        next_save_model = SAVE_MODEL_FREQUENCY

        while True:
            # Run multiple games in parallel
            future_games = [executor.submit(play_game, policy_net, epsilon) for _ in range(NUM_GAMES_PARALLEL)]

            for future in concurrent.futures.as_completed(future_games):
                new_experiences = future.result()

                # Safely add experiences to shared buffer
                with buffer_lock:
                    shared_exp_buffer.extend(new_experiences)

                    # Maintain fixed buffer size
                    while len(shared_exp_buffer) > MEMORY_SIZE:
                        shared_exp_buffer.pop(0)

            game_counter += NUM_GAMES_PARALLEL

            # Logging
            if game_counter >= next_log:
                logger.info(f"{game_counter} games played. Buffer size: {len(shared_exp_buffer)}")
                writer.add_scalar("epsilon", epsilon, game_counter)
                writer.add_scalar("buffer_size", len(shared_exp_buffer), game_counter)
                next_log += LOG_FREQUENCY

            if game_counter >= next_train:
                train_dqn()
                next_train += TRAIN_FREQUENCY

            if game_counter >= next_target_update:
                target_net.load_state_dict(policy_net.state_dict())
                next_target_update += TARGET_UPDATE_FREQUENCY

            if game_counter >= next_save_buffer:
                save_replay_buffer_torch(shared_exp_buffer, f"./replay_buffers/buffer-{timestamp}.pth")
                next_save_buffer += SAVE_BUFFER_FREQUENCY
            
            if game_counter >= next_save_model:
                torch.save(policy_net.state_dict(), f"./models/model-{timestamp}.pth")
                next_save_model += SAVE_MODEL_FREQUENCY

            # Epsilon decay
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY * game_counter)

            # Stopping condition
            if len(shared_exp_buffer) >= MEMORY_SIZE:
                break

# Save final model and buffer
writer.close()
torch.save(
    policy_net.state_dict(),
    f"./models/final-model-{timestamp}.pth"    
)
save_replay_buffer_torch(
    shared_exp_buffer,
    f"./replay_buffers/final-buffer-{timestamp}.pth"
)