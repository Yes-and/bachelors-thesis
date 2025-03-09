import json
import logging
import os

import numpy as np

from pyminion.core import Card
from pyminion.game import Game
from pyminion.player import Player
from pyminion.result import GameResult

from src.data_structures import Experience



class CustomGame(Game):
    """
    Modified version of the game class.

    This version makes it easier to collect important information.
    """
    def __init__(
        self,
        players: list[Player],
        expansions: list[list[Card]],
        exp_buffer: list[Experience],
        log_stdout: bool
    ):
        super().__init__(
            players=players,
            expansions=expansions,
            log_stdout=log_stdout
        )
        self.exp_buffer = exp_buffer
        self.player_VPs = 0
        self.enemy_VPs = 0
        self.last_state = None
        self.last_action = None
        self.max_turns = 50

    def play(self) -> GameResult:
        self.start()
        while True:
            for player in self.players:
                self.current_player = player
                self.play_turn(player)

                if player.turns >= self.max_turns:
                    self.is_over = True
                    result = self.summarize_game()
                    logging.info(f"Game was stopped after {self.max_turns} turns, \n{result}")
                    return result

                if player.player_id == 'my_bot':
                    state = player.state_before_action
                    action = player.decider.last_action
                    reward = player.decider.last_reward
                    done = False
                    new_state = player.state_after_action

                    self.last_state = new_state
                    self.last_action = action

                    current_experience = Experience(
                        state, action, reward, done, new_state
                    )
                    self.exp_buffer.append(current_experience)

                if self.is_over():
                    state = self.last_state
                    action = self.last_action
                    
                    # Get final reward
                    i = [str(player) for player in self.players].index('my_bot')
                    my_bot = self.players[i]
                    my_bot.decider.set_turn_reward(
                        player=my_bot,
                        game=self
                    )
                    # reward = my_bot.decider.last_reward
                    turns = my_bot.turns
                    self.player_VPs = self.players[i].get_victory_points()
                    self.enemy_VPs = self.players[1-i].get_victory_points()
                    denominator = (self.player_VPs + self.enemy_VPs)
                    if denominator == 0: # Prevent division by zero
                        denominator = 1
                    final_reward = (self.player_VPs - self.enemy_VPs - 0.5) / (denominator) / np.sqrt(turns)
                    if turns >= self.max_turns: # Prevent the player not doing anything and still winning
                        final_reward = -1
                    done = True

                    # Get new state
                    i = [str(player) for player in self.players].index('my_bot')
                    self.players[i].decider.set_current_state(
                        player=self.players[i],
                        game=self
                    )
                    new_state = self.players[i].decider.last_state


                    current_experience = Experience(
                        state, action, final_reward, done, new_state
                    )
                    self.exp_buffer.append(current_experience)

                    result = self.summarize_game()
                    logging.info(f"\n{result}")
                    return result
