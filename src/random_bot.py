from typing import Iterator
import random

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



class RandomBot(OptimizedBot):
    def __init__(
        self,
        player_id: str = "random_bot",
    ):
        super().__init__(decider=RandomBotDecider(), player_id=player_id)

class RandomBotDecider(OptimizedBotDecider):
    """
    Does random things.
    """
    def __init__(
        self
    ):
        super().__init__()
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
            17: mine
        }

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

    def buy_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        available_choices = [card for card in self.action_mapping.values() if card.base_cost.money < player.state.money]
        chosen_card = random.choice(available_choices+[None])
        if chosen_card is not None:
            return iter([chosen_card])
        else:
            return iter([])
