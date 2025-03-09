import copy

from src.actor_critic_model import ActorCritic
from src.data_structures import ExperienceMemory
from src.game import CustomGame
from src.my_bot import MyBot
from src.sets import custom_set

from pyminion.bots.examples import BigMoney
import numpy as np
import torch



# Hyperparameters
GAMMA = 0.95 # Discount factor proposed by ChatGPT because Dominion has long-term rewards
LR = 1e-5 # Originally 0.001 (lower)
STATE_SIZE = 38
ACTION_SIZE = 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load policy net inside the subprocess
policy_net = ActorCritic(STATE_SIZE, ACTION_SIZE).to(DEVICE)
policy_net.eval()

bot1 = BigMoney()
bot2 = MyBot(net=policy_net)
game = CustomGame(
    players=[bot1, bot2],
    expansions=[custom_set],
    log_stdout=True,
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

player_VPs = game.player_vp
enemy_VPs = game.enemy_vp
