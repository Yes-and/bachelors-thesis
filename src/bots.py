from typing import Iterator
import json
import os
import random

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
        epsilon=0.0,
        player_id: str = "my_bot"
    ):
        super().__init__(
            decider=MyBotDecider(
                net, epsilon
            ), 
            player_id=player_id
            )

class MyBotDecider(OptimizedBotDecider):
    def __init__(
        self,
        net,
        epsilon
    ):
        super().__init__()
        self.net = net
        self.epsilon = epsilon

    def action_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        """
        The action priority is dictated by a simple heuristic:
        1. Use cards that give actions
        2. Use cards that give a large advantage
        3. Use cards that give more cards
        4. Use all other cards
        """
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

    def play_step(self, state):
        if np.random.random() < self.epsilon:
            action = random.randint(1, 18)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(state_tensor)
            action = torch.argmax(q_values).item() + 1

        return action

    def buy_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        money = player.state.money
        buys = player.state.buys

        # Game state
        pile_cards = {str(pile.name): len(pile) for pile in game.supply.piles}
        
        # Player state
        player_cards = {name: 0 for name in pile_cards.keys()}
        for card in player.get_all_cards():
            player_cards[str(card)] += 1

        complete_state = list(pile_cards.values()) + \
            list(player_cards.values()) + \
            [money] + [buys]
        indexes = ['pile_' + val for val in list(pile_cards.keys())] + \
            ['player_' + val for val in list(player_cards.keys())] + \
            ['player_money', 'player_buys']
        if os.environ.get('INDEXES'):
            if not (json.loads(os.environ['INDEXES'])==indexes):
                raise Exception('The index order has changed!')
        else:
            os.environ['INDEXES'] = json.dumps(indexes)
        os.environ['STATE'] = json.dumps(complete_state)

        action_code = self.play_step(state = complete_state)
        os.environ['ACTION'] = str(action_code)

        action_mapping = {
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
        card = action_mapping[action_code]
        os.environ['REWARD'] = str(0.01)
        if not card:
            return iter([])
        elif (money >= card.base_cost.money):
            return iter(
                [action_mapping[action_code]]
            )
        else:
            print('Illegal buy decision was made.')
            os.environ['REWARD'] = str(-1.0)
            return iter([])
