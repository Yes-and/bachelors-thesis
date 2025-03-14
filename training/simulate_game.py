import torch
from pyminion.bots.examples import BigMoney

from src.actor_critic_model import ActorCritic
from game_logic.game import CustomGame
from game_logic.sets import custom_set
from game_logic.bots.my_bot import MyBot



def play_game(g):
    """Runs a game using a locally loaded PPO policy network and returns an experience buffer."""
    # Load policy net inside the subprocess
    policy_net = ActorCritic(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    policy_net.load_state_dict(torch.load(g.MODEL_PATH, map_location=g.DEVICE)) # Load the network
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

    updated_exp_buffer = game.exp_buffer

    # Variables for monte carlo returns
    returns = []
    cumulative_reward = 0

    # Propagate backward through the trajectory
    for i in reversed(range(len(game.exp_buffer.rewards))):
        cumulative_reward = game.exp_buffer.rewards[i] + g.GAMMA * cumulative_reward  # Apply discount factor
        returns.insert(0, cumulative_reward)  # Store discounted return

    # Replace reward with monte carlo returns
    updated_exp_buffer.rewards = returns

    # Extract information regarding the number of victory points
    player_vp = game.player_vp
    enemy_vp = game.enemy_vp
    turns = game.player_turns

    return updated_exp_buffer, player_vp, enemy_vp, turns
