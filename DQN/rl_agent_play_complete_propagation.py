import datetime
import logging
import multiprocessing
import os
import concurrent.futures

from pyminion.bots.examples import BigMoney

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
# from collections import deque

from src.bots import MyBot, RandomBot
from src.data_structures import Experience
from src.dqn_model import DQN
from src.game import CustomGame
from src.sets import custom_set
from src.tools import (
    save_replay_buffer_torch, 
    load_replay_buffer_torch,
    sample_experience_batch
)



# Hyperparameters
GAMMA = 0.95 # Discount factor that should prioritize long term rewards
LR = 1e-5 # Originally 0.001 (lower)

INPUT_SHAPE = 38
N_ACTIONS = 18

BATCH_SIZE = 128 # Originally 64, this should make it smoother
MEMORY_SIZE = 10000 # Training should stop at that point. Originally 50k. (Change to 10k and drop old experiences)
MAX_EXPERIENCES = 100000

EPSILON_START = 1 # Full exploration in the beginning
EPSILON_END = 0.1
EPSILON_DECAY = 0.0005 # Originally 0.002, then 0.005

LOG_FREQUENCY = 10
NUM_GAMES_PARALLEL = 6
TRAIN_FREQUENCY = 15 # Train network every 450 experiences ~= 15 games
TARGET_UPDATE_FREQUENCY = 25 # Originally 50
N_EPOCHS = 2 # Number of epochs for each training round
SAVE_BUFFER_FREQUENCY = 250
SAVE_MODEL_FREQUENCY = 250
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./models/temp-policy-net.pth"  # Temp file for policy network



def train_dqn(policy_net, target_net, optimizer, loss_fn, shared_exp_buffer):
    """Train the DQN using sampled experience replay."""
    if len(shared_exp_buffer) < MEMORY_SIZE:
        return 0, 0, 0 # Skip training if insufficient experiences
    
    batch = sample_experience_batch(list(shared_exp_buffer), BATCH_SIZE)
    states, actions, rewards, dones, next_states = batch

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)

    # Train for 3 epochs
    for _ in range(N_EPOCHS):
        # Compute Q-values for the actions taken
        q_values = policy_net(states).gather(1, actions - 1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            # next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards # + (GAMMA * next_q_values * (1 - dones))

        # Compute loss and optimize
        loss = loss_fn(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10) # Add gradient clipping to prevent exploding Q-values
        optimizer.step()

    num_q_vals = q_values.detach().numpy()
    return loss, np.mean(num_q_vals), np.std(num_q_vals)



def play_game(epsilon):
    """Runs a game with a locally loaded policy network and returns a local experience buffer."""
    local_exp_buffer = list()

    # Load policy net inside the subprocess
    policy_net = DQN(INPUT_SHAPE, N_ACTIONS).to(DEVICE)
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    policy_net.eval()

    bot1 = RandomBot()
    bot2 = MyBot(net=policy_net, epsilon=epsilon)  # Use locally loaded network
    game = CustomGame(
        players=[bot1, bot2],
        expansions=[custom_set],
        exp_buffer=local_exp_buffer,
        log_stdout=False,
    )
    game.play()  # Fills local_exp_buffer

    discounted_return = 0  # Initialize the return value

    # Iterate backward and update in place
    for i in reversed(range(len(local_exp_buffer))):
        exp = local_exp_buffer[i]

        if exp.done:
            discounted_return = 0  # Reset return at episode end

        # Compute the updated return
        discounted_return = exp.reward + GAMMA * discounted_return

        # Update the experience in place
        local_exp_buffer[i] = Experience(
            state=exp.state,
            action=exp.action,
            reward=discounted_return,  # Store updated reward
            done=exp.done,
            new_state=exp.new_state
        )

    player_VPs = game.player_VPs
    enemy_VPs = game.enemy_VPs

    return list(local_exp_buffer), player_VPs, enemy_VPs



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize Logging
    timestamp = str(datetime.datetime.now())[:19].replace(":", "-")
    logger = logging.getLogger('specific_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./logs/run-{timestamp}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Training DQN while playing against BigMoney")
    logger.info(f"GAMMA: {GAMMA}")
    logger.info(f"LR: {LR}") 
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"MEMORY_SIZE: {MEMORY_SIZE}")
    logger.info(f"TRAIN_FREQUENCY: {TRAIN_FREQUENCY}")
    logger.info(f"TARGET_UPDATE_FREQUENCY: {TARGET_UPDATE_FREQUENCY}")

    # Initialize Networks and Training Components
    policy_net = DQN(INPUT_SHAPE, N_ACTIONS).to(DEVICE)
    # policy_net.load_state_dict(torch.load("./models/model-2025-03-07 23-10-22.pth", map_location=DEVICE)) # Load the network
    target_net = DQN(INPUT_SHAPE, N_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter()

    # Experience buffer
    exp_buffer = list()
    # exp_buffer.extend(
    #     load_replay_buffer_torch('./replay_buffers/buffer-2025-03-07 23-10-22.pth')
    # )
    logger.info(f"Initial replay buffer size: {len(exp_buffer)}")

    epsilon = EPSILON_START
    game_counter = 1
    total_experiences = 0

    # Save initial policy net for subprocesses
    torch.save(policy_net.state_dict(), MODEL_PATH)

    # Training Loop
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GAMES_PARALLEL) as executor:
        next_log, next_train, next_target_update = LOG_FREQUENCY, TRAIN_FREQUENCY, TARGET_UPDATE_FREQUENCY
        next_save_buffer, next_save_model = SAVE_BUFFER_FREQUENCY, SAVE_MODEL_FREQUENCY

        while True:
            # Run multiple games in parallel
            futures = [executor.submit(play_game, epsilon) for _ in range(NUM_GAMES_PARALLEL)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    new_experiences, player_VPs, enemy_VPs = future.result(timeout=10)
                    if (new_experiences and player_VPs and enemy_VPs):
                        exp_buffer.extend(new_experiences)
                        total_experiences += len(new_experiences)
                        writer.add_scalar("player_victory_points", player_VPs, game_counter)
                        writer.add_scalar("enemy_victory_points", enemy_VPs, game_counter)
                    else:
                        logger.info("Worker returned no experiences")
                    # Maintain buffer size (FIFO strategy)
                    if len(exp_buffer) > MEMORY_SIZE:
                        exp_buffer = exp_buffer[-MEMORY_SIZE:]
                except concurrent.futures.TimeoutError:
                    logger.info("Warning: a worker took too long! Terminating...")
                except Exception as e:
                    logger.info(f"Worker process failed: {e}")

            game_counter += NUM_GAMES_PARALLEL

            # Logging
            if game_counter >= next_log:
                logger.info(f"{game_counter} games played. Buffer size: {len(exp_buffer)}. Total experiences: {total_experiences}")
                writer.add_scalar("epsilon", epsilon, game_counter)
                writer.add_scalar("buffer_size", len(exp_buffer), game_counter)
                next_log += LOG_FREQUENCY

            if game_counter >= next_train:
                logger.info("Fitting the DQN.")
                loss, mean_q, std_q = train_dqn(policy_net, target_net, optimizer, loss_fn, exp_buffer)
                logger.info("The DQN has been fitted.")
                writer.add_scalar("loss", loss, game_counter)
                writer.add_scalar("mean_q_values", mean_q, game_counter)
                writer.add_scalar("std_q_values", std_q, game_counter)
                next_train += TRAIN_FREQUENCY

            if game_counter >= next_target_update:
                logger.info("Updating target network.")
                target_net.load_state_dict(policy_net.state_dict())
                logger.info("Target network has been updated.")
                next_target_update += TARGET_UPDATE_FREQUENCY
                torch.save(policy_net.state_dict(), MODEL_PATH)  # Update model for subprocesses

            if game_counter >= next_save_buffer:
                logger.info("Saving replay buffer.")
                save_replay_buffer_torch(exp_buffer, f"./replay_buffers/buffer-{timestamp}.pth")
                logger.info("Replay buffer has been saved.")
                next_save_buffer += SAVE_BUFFER_FREQUENCY

            if game_counter >= next_save_model:
                logger.info("Saving policy network model.")
                torch.save(policy_net.state_dict(), f"./models/model-{timestamp}.pth")
                logger.info("Policy network has been saved.")
                next_save_model += SAVE_MODEL_FREQUENCY

            # Epsilon decay
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY * game_counter)

            if total_experiences > MAX_EXPERIENCES:
                break

    writer.close()    # exp_buffer.extend(
    #     load_replay_buffer_torch('./replay_buffers/buffer-2025-03-07 23-10-22.pth')
    # )
    torch.save(policy_net.state_dict(), f"./models/final-model-{timestamp}.pth")
    save_replay_buffer_torch(exp_buffer, f"./replay_buffers/final-buffer-{timestamp}.pth")

    # Cleanup temp model file
    os.remove(MODEL_PATH)
