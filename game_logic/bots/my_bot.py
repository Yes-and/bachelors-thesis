import random
from typing import Iterator

import numpy as np
import torch

from pyminion.bots.optimized_bot import OptimizedBotDecider
from pyminion.bots.optimized_bot import OptimizedBot
from pyminion.core import Card
from pyminion.expansions.base import (
    estate,
    duchy,
    province,
    curse,
    copper,
    silver,
    gold,
    cellar,
    market,
    merchant,
    militia,
    mine,
    moat,
    remodel,
    smithy,
    village,
    workshop
)
from pyminion.game import Game



class MyBot(OptimizedBot):
    def __init__(
        self,
        net,
        prob_action,
        memory,
        player_id: str = "my_bot"
    ):
        super().__init__(
            decider=MyBotDecider(),
            player_id=player_id
            )
        self.net = net
        self.prob_action = prob_action
        self.memory = memory

    def take_turn(self, game: "Game", is_extra_turn: bool = False) -> None:
        self.start_turn(game, is_extra_turn)
        self.start_action_phase(game)
        self.start_treasure_phase(game)
        
        # Save game states before action
        self.decider.set_current_state(self, game)
        self.start_buy_phase(game)
        self.decider.reset_state() # Reset the state for debugging

        self.start_cleanup_phase(game)
        self.end_turn(game)

    def get_good_cards_ratio(self, game: "Game") -> float:
        all_cards = [card for card in self.get_all_cards()]
        non_trash_cards = [card for card in all_cards if card not in [curse, copper, estate]]
        return len(non_trash_cards) / len(all_cards)

class MyBotDecider(OptimizedBotDecider):
    def __init__(
        self
    ):
        super().__init__()
        self.state = None
        self.action = None
        self.turn = None
        self.state_indexes = {}
        self.action_mapping = {
            1: estate,
            2: duchy,
            3: province,
            4: curse,
            5: copper,
            6: silver,
            7: gold,
            8: cellar,
            9: moat,
            10: merchant,
            11: village,
            12: workshop,
            13: militia,
            14: remodel,
            15: smithy,
            16: market,
            17: mine,
            18: None
        }

    def set_current_state(self, player: "Player", game: "Game"):
        """
        Sets a normalized state to the MyBotDecider object, so that it can be retrieved later.
        """
        # Game state
        pile_cards = {str(pile.name): (len(pile) if pile in game.supply.piles else 0) for pile in game.supply.piles}

        # Player state
        player_cards = {name: 0 for name in pile_cards.keys()}
        for card in player.get_all_cards():
            player_cards[str(card)] += 1

        # Totals for division (so that state values are normalized)
        pile_cards_div = [8, 8, 8, 10, 46, 40, 30, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        player_cards_div = pile_cards_div

        divs = pile_cards_div + \
            player_cards_div
        complete_state = list(pile_cards.values()) + \
            list(player_cards.values())
        normalized_state = np.array(complete_state) / np.array(divs)
        self.state = normalized_state

        state_indexes = ['pile_' + val for val in list(pile_cards.keys())] + \
            ['player_money', 'player_discard_size'] + \
            ['player_' + val for val in list(player_cards.keys())] + \
            ['enemy_money', 'enemy_VPs']
        if not self.state_indexes:
            self.state_indexes = state_indexes
        else:
            if not (self.state_indexes==state_indexes):
                raise Exception('The order of cards in the state has changed!')

    def reset_state(self):
        """
        Resets the state for easier debugging.
        """
        self.state = None

    def action_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        """
        Currently no action are used for simplification.
        """
        # """
        # The action priority is dictated by a simple heuristic:
        # 1. Use cards that give actions
        # 2. Use cards that give a large advantage
        # 3. Use cards that give more cards
        # 4. Use all other cards
        # """
        while (player.state.actions > 0):
            if village in player.hand.cards:
                yield village
            elif market in player.hand.cards:
                yield market
            elif merchant in player.hand.cards:
                yield merchant
            elif cellar in player.hand.cards:
                yield cellar
            elif mine in player.hand.cards:
                yield mine
            elif militia in player.hand.cards:
                yield militia
            elif smithy in player.hand.cards:
                yield smithy
            elif remodel in player.hand.cards:
                yield remodel
            elif workshop in player.hand.cards:
                yield workshop
            elif moat in player.hand.cards:
                yield moat
            else:
                return iter([])

    @torch.no_grad()
    def buy_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        # Filter valid actions
        valid_actions = [1] * 18
        cards = list(self.action_mapping.values())[:17]
        for i in range(17):
            if cards[i].base_cost.money > player.state.money:
                valid_actions[i] = 0

        if (not self.turn) and (random.random() < player.prob_action):
            valid_actions = [i for i in range(17) if valid_actions[i]==1] # Convert to a numerical list
            self.action = random.choice(valid_actions)
            self.turn = player.turns

            # Save important variables
            player.memory.state = self.state
            player.memory.action = self.action
            player.memory.turn = self.turn
        else:
            # Get the game state
            state_tensor = torch.tensor(
                self.state,
                dtype=torch.float32
            ).unsqueeze(0)

            # Get valid action mask (1 for valid, 0 for invalid)
            valid_action_mask = torch.tensor([valid_actions])

            action, log_prob, probs = player.net.get_action(state_tensor, valid_action_mask)
            
            self.action = action

        # Set return card
        card = self.action_mapping[self.action+1]

        if not card:
            return iter([])
        else:
            return iter([card])
