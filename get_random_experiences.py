from collections import deque
import logging

import numpy as np
from pyminion.bots.examples import BigMoney

from src.bots import MyBot
from src.dqn_model import DQN
from src.game import CustomGame
from src.sets import custom_set
from src.tools import save_replay_buffer_torch



# Hyperparameters
INPUT_SHAPE = 38
N_ACTIONS = 18
REPLAY_BUFFER_SIZE = 10000
EPSILON = 1.0

# Set up logging
logger = logging.getLogger('specific_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('./replay_buffers/random_replay_buffer_progress.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

policy_net = DQN(
    input_dim=INPUT_SHAPE,
    output_dim=N_ACTIONS
)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

logger.info('Generating random training data.')
n_games = 0
turns = []

while True:
    if len(replay_buffer)==REPLAY_BUFFER_SIZE:
        logger.info(f'All {REPLAY_BUFFER_SIZE} training samples have been created.')
        logger.info(f'A total of {n_games} games have been played.')
        logger.info(f'On average, there were {round(np.mean(turns))} turns, minimum of {min(turns)} turns and a maximum of {max(turns)} turns.')
        logger.info('Saving replay buffer')
        save_replay_buffer_torch(
            replay_buffer, 
            filename='./replay_buffers/random_replay_buffer.pth'
        )
        break

    n_games += 1
    logger.info(f'There are currently {len(replay_buffer)} experiences in the replay buffer.')

    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net, epsilon=EPSILON)

    game = CustomGame(
        players=[bot1, bot2],
        expansions=[custom_set],
        exp_buffer=replay_buffer
    )
    result = game.play()
    
    turns.append(bot1.turns)
