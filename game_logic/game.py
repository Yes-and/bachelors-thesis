import logging

from pyminion.core import Card
from pyminion.game import Game
from pyminion.player import Player
from pyminion.result import GameResult

from game_logic.data_structures import GameExperienceMemory



class CustomGame(Game):
    """
    Modified version of the game class.

    This version makes it easier to collect important information.
    """
    def __init__(
        self,
        players: list[Player],
        expansions: list[list[Card]],
        log_stdout: bool
    ):
        super().__init__(
            players=players,
            expansions=expansions,
            log_stdout=log_stdout
        )
        self.exp_buffer = GameExperienceMemory()
        self.player_vp = 0
        self.enemy_vp = 0
        self.player_turns = 0
        self.max_turns = 50

    def play(self) -> GameResult:
        self.start()
        while True:
            for player in self.players:
                self.current_player = player

                # Stop the game if max turns is reached or if the game is over
                if (player.turns >= self.max_turns):
                    # Summarize the game for debugging
                    result = self.summarize_game()
                    logging.info(f"\n{result}")
                    
                    # Get reward
                    i = [str(player) for player in self.players].index('my_bot')
                    self.player_vp = self.players[i].get_victory_points()
                    self.enemy_vp = self.players[1-i].get_victory_points()
                    self.player_turns = self.players[i].turns
                    
                    # We don't want the agent to play long games
                    reward = -1

                    # Save reward to last experience
                    self.exp_buffer.rewards[-1] = reward

                    return result
                    
                
                if (self.is_over()):
                    # Summarize the game for debugging
                    result = self.summarize_game()
                    logging.info(f"\n{result}")

                    # Get reward
                    i = [str(player) for player in self.players].index('my_bot')
                    self.player_vp = self.players[i].get_victory_points()
                    self.enemy_vp = self.players[1-i].get_victory_points()
                    self.player_turns = self.players[i].turns
                    reward = 5 if (self.player_vp>self.enemy_vp) else -5

                    # Save reward to last experience
                    self.exp_buffer.rewards[-1] = reward

                    return result

                self.play_turn(player)

                if player.player_id == 'my_bot':
                    state = player.state_before_action
                    action = player.decider.action
                    state_value = player.decider.state_value
                    log_prob = player.decider.log_prob
                    reward = player.decider.reward
                    done = False

                    # Save experiences to exp buffer
                    self.exp_buffer.store(
                        state=state,
                        action=action,
                        state_value=state_value,
                        log_prob=log_prob,
                        reward=reward,
                        done=done
                    )
