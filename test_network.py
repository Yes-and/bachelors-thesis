import copy

from src.network_model import PolicyNetwork
from game_logic.data_structures import GameExperienceMemory
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

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/model-2025-03-22 20-11-50.pth"

g = GlobalVariables()

policy_net = PolicyNetwork(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE)) # Load the network
policy_net.eval() # Sets network to evaluation mode

memory = GameExperienceMemory()

# Set up the game
bot1 = BigMoney()
bot2 = MyBot(net=policy_net, prob_action=1/25, memory=memory)
game = CustomGame(
    players=[bot1, bot2],
    expansions=[custom_set],
    log_stdout=True,
    memory=memory
)
game.play()  # Fills game.exp_buffer

if memory.action is None:
    memory.action = 18
    memory.turn = 1
    memory.state = np.array( # Starting state
        [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.375,0.,0.,0.,0.15217391,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    )
