import copy

from src.actor_critic_model import ActorCritic
from game_logic.game import CustomGame
from game_logic.bots.my_bot import MyBot
from game_logic.sets import custom_set

import numpy as np
from pyminion.bots.examples import BigMoney
import torch



class GlobalVariables:
    # State representation
    STATE_SIZE = 34
    ACTION_SIZE = 18

    # Hyperparameters
    GAMMA = 0.99  # Discount factor
    LAMBDA = 0.95 # Bias-variance tradeoff factor
    LR = 1e-5  # Learning rate
    EPSILON_CLIP = 0.2  # PPO Clipping range
    EPOCHS = 10  # Number of tratates, actions, loning epochs per batch
    BATCH_SIZE = 64  # Mini-batch size for training
    EPISODES = 5000  # Training episodes

    # Others
    NUM_GAMES_PARALLEL = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/temp-policy-net.pth"

    # Logging
    SAVE_MODEL_FREQUENCY = 200
    LOG_FREQUENCY = 100

g = GlobalVariables()

policy_net = ActorCritic(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
# policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE)) # Load the network
policy_net.eval() # Sets network to evaluation mode

# Set up the game
bot1 = BigMoney()
bot2 = MyBot(net=policy_net)
game = CustomGame(
    players=[bot1, bot2],
    expansions=[custom_set],
    log_stdout=False,
)
game.play()  # Fills game.exp_buffer

exp = game.exp_buffer

# Generalized advantage estimation
advantages = np.zeros(len(exp.rewards))  # Initialize advantage array
last_advantage = 0

for i in reversed(range(len(exp.rewards))):
    # Compute TD error (delta)
    delta = exp.rewards[i] + g.GAMMA * (exp.state_values[i + 1] if i < len(exp.rewards) - 1 else 0) - exp.state_values[i]
    
    # Compute GAE advantage
    advantages[i] = last_advantage = delta + g.GAMMA * g.LAMBDA * last_advantage

# Normalize advantages and add them to the buffer
exp.advantages = advantages

# Extract information regarding the number of victory points
player_vp = game.player_vp
enemy_vp = game.enemy_vp
turns = game.player_turns
