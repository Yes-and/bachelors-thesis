import logging

from pyminion.core import Card
from pyminion.game import Game
from pyminion.player import Player
from pyminion.result import GameResult

from game_logic.data_structures import GameExperienceMemory



class CustomGame(Game):
    """
    Modified version of the game class.

    This version makes it easier to collect important information
    and pass the game memory to the bot used by the agent.
    """
    def __init__(
        self,
        players: list[Player],
        expansions: list[list[Card]],
        log_stdout: bool,
        memory: GameExperienceMemory
    ):
        super().__init__(
            players=players,
            expansions=expansions,
            log_stdout=log_stdout
        )
        self.memory = memory
        self.player_vp = None
        self.enemy_vp = None
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

                    # Add negative reward for stretching the game
                    self.memory.end_turns = 50
                    self.memory.end_reward = -1

                    return result
                
                if (self.is_over()):
                    # Summarize the game for debugging
                    result = self.summarize_game()
                    logging.info(f"\n{result}")

                    # Calculate the end reward
                    i = [str(player) for player in self.players].index('my_bot')
                    self.player_vp = self.players[i].get_victory_points()
                    self.enemy_vp = self.players[1-i].get_victory_points()
                    
                    good_cards_ratio = self.players[i].get_good_cards_ratio()
                    end_reward = (self.player_vp - self.enemy_vp) / 50 + good_cards_ratio 
                    # TODO: add more complex reward structure

                    self.memory.end_turns = self.players[i].turns
                    self.memory.end_reward = end_reward

                    return result

                self.play_turn(player)
