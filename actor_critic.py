import datetime
import logging
import multiprocessing
import os
import concurrent.futures

from src.actor_critic_model import ActorCritic
from src.data_structures import ExperienceMemory
from src.game import CustomGame
from src.my_bot import MyBot
from src.sets import custom_set

from pyminion.bots.examples import BigMoney
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 0.0003  # Learning rate
EPSILON_CLIP = 0.2  # PPO Clipping range
EPOCHS = 5  # Number of training epochs per batch
BATCH_SIZE = 32  # Mini-batch size for training
EPISODES = 5000  # Training episodes

STATE_SIZE = 38
ACTION_SIZE = 18

NUM_GAMES_PARALLEL = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/temp-policy-net.pth"  # Temp file for policy network

LOG_FREQUENCY = 60
SAVE_MODEL_FREQUENCY = 200



def play_game():
    """Runs a game using a locally loaded PPO policy network and returns an experience buffer."""
    # Load policy net inside the subprocess
    policy_net = ActorCritic(STATE_SIZE, ACTION_SIZE).to(DEVICE)
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # Load the network
    policy_net.eval()

    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net)
    game = CustomGame(
        players=[bot1, bot2],
        expansions=[custom_set],
        log_stdout=False,
    )
    game.play()  # Fills local_exp_buffer

    updated_exp_buffer = game.exp_buffer

    returns = []
    G = 0

    # Propagate backward through the trajectory
    for i in reversed(range(len(game.exp_buffer.rewards))):
        G = game.exp_buffer.rewards[i] + GAMMA * G  # Apply discount factor
        returns.insert(0, G)  # Store discounted return

    updated_exp_buffer.rewards = returns

    player_vp = game.player_vp
    enemy_vp = game.enemy_vp

    return updated_exp_buffer, player_vp, enemy_vp

def train_policy(policy_net, exp_buffer):
    """Trains the PPO policy network using the collected experience buffer."""

    if len(exp_buffer.states) < BATCH_SIZE:
        logger.info("Not enough experiences to train yet.")
        return

    # Convert experience buffer to tensors
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in list(exp_buffer.states)]).to(DEVICE)
    actions = torch.tensor(list(exp_buffer.actions), dtype=torch.long, device=DEVICE)
    old_log_probs = torch.stack([torch.tensor(lp, dtype=torch.float32) for lp in list(exp_buffer.log_probs)]).to(DEVICE)
    returns = torch.tensor(list(exp_buffer.rewards), dtype=torch.float32, device=DEVICE)

    # Compute advantage
    values = policy_net(states)[1].squeeze()
    advantages = returns - values.detach()

    # PPO Training (Multiple epochs per batch)
    for _ in range(EPOCHS):
        new_action_probs, new_values = policy_net(states)
        new_action_log_probs = torch.log(new_action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Compute PPO ratio
        ratio = torch.exp(new_action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP)

        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns)

        loss = actor_loss + 0.5 * critic_loss  # Balance actor and critic losses

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(policy_net.state_dict(), MODEL_PATH)

    return loss.item()

# Initialize environment and model
policy_net = ActorCritic(STATE_SIZE, ACTION_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

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
    writer = SummaryWriter()

    # Log hyperparameters
    logger.info("Training PPO while playing against BigMoney")
    logger.info(f"GAMMA: {GAMMA}")
    logger.info(f"LR: {LR}")
    logger.info(f"EPSILON CLIP: {EPSILON_CLIP}")
    logger.info(f"EPOCHS: {EPOCHS}")
    logger.info(f"BATCH SIZE: {BATCH_SIZE}")

    # Initialize the actor-critic network
    policy_net = ActorCritic(STATE_SIZE, ACTION_SIZE).to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    torch.save(policy_net.state_dict(), MODEL_PATH)  # Save initial policy

    # Shared experience memory
    exp_buffer = ExperienceMemory()
    game_counter = 1

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GAMES_PARALLEL) as executor:
        next_log, next_train, next_save_model = LOG_FREQUENCY, BATCH_SIZE, SAVE_MODEL_FREQUENCY

        while True:
            # Start multiple games in parallel
            futures = [executor.submit(play_game) for _ in range(NUM_GAMES_PARALLEL)]

            # Wait for all workers to complete before continuing
            completed, _ = concurrent.futures.wait(futures)

            for future in completed:
                try:
                    new_experiences, player_vp, enemy_vp = future.result(timeout=10)
                    if new_experiences is not None:
                        exp_buffer.store_batch(new_experiences)  # Store experiences in batch
                        writer.add_scalar("victory_points_player", player_vp, game_counter)
                        writer.add_scalar("victory_points_enemy", enemy_vp, game_counter)
                        logger.info(f"Game {game_counter} completed. Player VP: {player_vp}, Enemy VP: {enemy_vp}")
                        game_counter += 1
                    else:
                        logger.warning("Worker returned no experiences.")
                except concurrent.futures.TimeoutError:
                    logger.error("Game execution timed out.")
                except Exception as e:
                    logger.error(f"Exception in worker: {str(e)}")

            # Log at regular intervals
            if game_counter >= next_log:
                logger.info(f"{game_counter} games have been played.")
                next_log += LOG_FREQUENCY

            # Train after collecting a full batch
            if game_counter >= next_train:
                logger.info(f"Collected {len(exp_buffer)} experiences. Training now...")
                loss = train_policy(policy_net, exp_buffer)
                logger.info("Policy network has been fitted. Clearing the buffer...")
                exp_buffer.clear()
                writer.add_scalar("loss", loss, game_counter)
                next_train += BATCH_SIZE

            # Save model periodically
            if game_counter >= next_save_model:
                logger.info("Saving policy network model.")
                torch.save(policy_net.state_dict(), f"./models/model-{timestamp}.pth")
                logger.info("Policy network has been saved.")
                next_save_model += SAVE_MODEL_FREQUENCY

            if game_counter >= EPISODES:
                break

    writer.close()
    torch.save(policy_net.state_dict(), f"./models/final-model-{timestamp}.pth")

    # Cleanup temp model file
    os.remove(MODEL_PATH)
