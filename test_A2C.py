import copy

from src.networks import A2C
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
    MODEL_PATH = "./selected_models/A2C-model.pth"

g = GlobalVariables()

shared_net = A2C(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
shared_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE)) # Load the network
shared_net.eval() # Sets network to evaluation mode

memory = GameExperienceMemory()

# Set up the game
bot1 = BigMoney()
bot2 = MyBot(net=shared_net, memory=memory, g=g)
game = CustomGame(
    players=[bot1, bot2],
    expansions=[custom_set],
    log_stdout=False,
    memory=memory
)
sim = Simulator(game, iterations=1000)
result = sim.run()
print(result)