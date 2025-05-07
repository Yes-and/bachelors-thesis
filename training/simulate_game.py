import random

from game_logic.data_structures import GameExperienceMemory, SingleExperienceMemory
from game_logic.game import CustomGame
from game_logic.bots.my_bot import MyBot
from game_logic.sets import custom_set
from src.networks import A2C, REINFORCE, DQN

from pyminion.bots.examples import BigMoney
import torch



def DQN_play_game(g):
    # Initialize a random DQN, load the temp model and set to evaluation mode
    policy_net = DQN(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE))
    policy_net.eval()

    # Prepare memory for the experiences
    memory = GameExperienceMemory()

    # Set up the game
    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net, memory=memory, g=g) # Bot used by the RL agent
    game = CustomGame(
        players=[bot1, bot2], # The order of the bots is different each game
        expansions=[custom_set],
        log_stdout=False,
        memory=memory
    )
    game.play()  # Fills memory

    # Sample an experience to use for training
    i = random.randint(1, memory.end_turns)-1

    # If the game ends on the last turn, the experience looks different
    # and uses the final reward instead of the intermediate one
    if i == (memory.end_turns-1):
        experience = SingleExperienceMemory(
            state = memory.states[i],
            action = memory.actions[i],
            reward = memory.end_reward,
            turn = i+1,
            end_turns = memory.end_turns,
            next_state = memory.states[i], # Not used in the TD equation
            done = True
        )
    else:
        experience = SingleExperienceMemory(
            state = memory.states[i],
            action = memory.actions[i],
            reward = memory.intermediate_rewards[i],
            turn = i+1,
            end_turns = memory.end_turns,
            next_state = memory.states[i+1],
            done = False
        )

    return experience, game.player_vp, game.enemy_vp

def A2C_play_game(g):
    # Initialize a random network for A2C, load state dict from temp model
    # and set the network to evaluation mode
    complex_net = A2C(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    complex_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE))
    complex_net.eval()

    # Prepare memory for experiences
    memory = GameExperienceMemory()

    # Set up the game
    bot1 = BigMoney()
    bot2 = MyBot(net=complex_net, memory=memory, g=g) # Bot used by RL agent
    game = CustomGame(
        players=[bot1, bot2], # The order of the bots is different each game
        expansions=[custom_set],
        log_stdout=False,
        memory=memory
    )
    game.play()  # Fills memory

    # Sample an experience to use for training
    i = random.randint(1, memory.end_turns)-1

    # If the game ends on the last turn, the experience looks different
    # and uses the final reward instead of the intermediate one because
    # unmodified A2C also uses TD errors
    if i == (memory.end_turns-1):
        experience = SingleExperienceMemory(
            state = memory.states[i],
            action = memory.actions[i],
            reward = memory.end_reward,
            turn = i+1,
            end_turns = memory.end_turns,
            next_state = memory.states[i], # Not be used in the TD equation
            done = True
        )
    else:
        experience = SingleExperienceMemory(
            state = memory.states[i],
            action = memory.actions[i],
            reward = memory.intermediate_rewards[i],
            turn = i+1,
            end_turns = memory.end_turns,
            next_state = memory.states[i+1],
            done = False
        )

    return experience, game.player_vp, game.enemy_vp

def REINFORCE_play_game(g):
    # Initialize a random REINFORCE network, load the state dict
    # of the temp network and set to evaluation mode
    policy_net = REINFORCE(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE))
    policy_net.eval()

    # Prepare memory for experiences
    memory = GameExperienceMemory()

    # Set up the game
    bot1 = BigMoney()
    bot2 = MyBot(net=policy_net, memory=memory, g=g) # Bot used by the RL agent
    game = CustomGame(
        players=[bot1, bot2], # The order of the bots is different each game
        expansions=[custom_set],
        log_stdout=False,
        memory=memory
    )
    game.play()  # Fills memory

    # Sample an experience to use for training
    # It is fairly simple because REINFORCE uses
    # Monte Carlo errors
    i = random.randint(1, memory.end_turns)-1
    experience = SingleExperienceMemory(
        state = memory.states[i],
        action = memory.actions[i],
        reward = memory.end_reward,
        turn = i+1,
        end_turns = memory.end_turns
    )

    return experience, game.player_vp, game.enemy_vp
