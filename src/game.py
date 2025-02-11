import json
import logging
import os

from pyminion.core import Card
from pyminion.game import Game
from pyminion.player import Player
from pyminion.result import GameResult

from src.data_structures import Experience, ExperienceBuffer



class CustomGame(Game):
    """
    Modified version of the game class.

    This version makes it easier to collect important information.
    """
    def __init__(
        self,
        players: list[Player],
        expansions: list[list[Card]],
        exp_buffer: ExperienceBuffer,
        kingdom_cards: list[Card]|None = None,
        start_deck: list[Card]|None = None,
        random_order: bool = True,
        log_stdout: bool = True,
        log_file: bool = False,
        log_file_name: str = "game.log",
    ):
        super().__init__(
            players,
            expansions,
            kingdom_cards,
            start_deck,
            random_order,
            log_stdout,
            log_file,
            log_file_name
        )
        self.exp_buffer = exp_buffer
        self.total_rewards = []
        self.game_rewards = []

    def play(self) -> GameResult:
        last_experience = None
        self.start()
        while True:
            for player in self.players:
                self.current_player = player
                self.play_turn(player)

                if self.current_player.player_id == 'my_bot':
                    # State after action
                    game_state = {str(pile.name): len(pile) for pile in self.supply.piles}
                    player_cards = [str(card) for card in self.current_player.get_all_cards()]
                    player_state = {card: 0 for card in game_state.keys()}
                    for card in player_cards:
                        player_state[card] += 1

                    state = json.loads(os.environ['STATE'])
                    action = int(os.environ['ACTION'])
                    reward = float(os.environ['REWARD'])
                    self.total_rewards.append(reward)
                    done = 0
                    new_state = None
                    current_experience = Experience(
                        state, action, reward, done, new_state
                    )
                    if not last_experience:
                        last_experience = current_experience
                    else:
                        last_experience = Experience(
                            last_experience[0],
                            last_experience[1],
                            last_experience[2],
                            last_experience[3],
                            current_experience[0]
                        )
                        self.exp_buffer.append(last_experience)

                if self.is_over():
                    money = self.current_player.state.money
                    buys = self.current_player.state.buys

                    # Game state
                    pile_cards = {str(pile.name): len(pile) for pile in self.supply.piles}

                    # Player state
                    player_cards = {name: 0 for name in pile_cards.keys()}
                    for card in self.current_player.get_all_cards():
                        player_cards[str(card)] += 1

                    winners = self.get_winners()
                    if ('my_bot' in [str(winner) for winner in winners]) \
                        and (len(winners)==1):
                        reward = 1.0
                    else:
                        reward = -1.0
                    self.total_rewards.append(reward)
                    self.game_rewards.append(reward)

                    done = 1

                    complete_state = list(pile_cards.values()) + \
                        list(player_cards.values()) + \
                        [money] + [buys]
                    
                    finishing_experience = Experience(
                        current_experience[0], current_experience[1], reward, done, complete_state
                    )
                    self.exp_buffer.append(finishing_experience)

                    result = self.summarize_game()
                    logging.info(f"\n{result}")
                    return result
