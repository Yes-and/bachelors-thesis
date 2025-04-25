import copy

from src.network_model import PolicyNetwork
from game_logic.data_structures import GameExperienceMemory
from game_logic.game import CustomGame
from game_logic.bots.my_bot import MyBot
from game_logic.sets import custom_set

import numpy as np
from pyminion.bots.examples import BigMoney
from pyminion.simulator import Simulator
import torch



class GlobalVariables:
    # State representation
    STATE_SIZE = 34
    ACTION_SIZE = 18

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/model-2025-04-20 15-20-16.pth"

g = GlobalVariables()

policy_net = PolicyNetwork(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE)) # Load the network
policy_net.eval() # Sets network to evaluation mode

memory = GameExperienceMemory()

# Set up the game
bot1 = BigMoney()
bot2 = MyBot(net=policy_net, prob_action=0, memory=memory)
game = CustomGame(
    players=[bot1, bot2],
    expansions=[custom_set],
    log_stdout=False,
    memory=memory
)
sim = Simulator(game, iterations=100)
result = sim.run()
print(result)