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
        memory,
        g,
        player_id: str = "my_bot"
    ):
        super().__init__(
            decider=MyBotDecider(),
            player_id=player_id
            )
        self.net = net
        self.memory = memory
        self.g = g

    def take_turn(self, game: "Game", is_extra_turn: bool = False) -> None:
        self.start_turn(game, is_extra_turn)

        self.start_action_phase(game)
        self.start_treasure_phase(game)
        
        # Save game states before action
        self.decider.set_current_state(self, game)
        self.start_buy_phase(game)
        self.decider.reset_state() # Reset the state for debugging

        # Set intermediate reward
        self.memory.intermediate_rewards.append(0) # TODO: implement intermediate reward calculation

        self.start_cleanup_phase(game)
        self.end_turn(game)

    def get_good_cards_ratio(self):
        all_cards = [card for card in self.get_all_cards()]
        good_cards = [card for card in all_cards if card not in [curse, copper, estate]] # These are judged as low value cards by the author
        ratio = len(good_cards) / len(all_cards)
        return ratio

class MyBotDecider(OptimizedBotDecider):
    def __init__(
        self
    ):
        super().__init__()
        self.state = None
        self.action = None
        self.turn = None
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
        self.state = normalized_state # Set the state as object variable

    def reset_state(self):
        """
        Resets the state for easier debugging.
        """
        self.state = None

    def action_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        """
        Define action priority for the decider used by the bot.
        It is currently as follows:
        1. Cards that give more actions are used
        2. Cards that upgrade low-value cards are used
        3. Cards that allow the player to draw more cards are used last
        Note: Remodel isn't used because it is complex.
        """
        while (player.state.actions > 0):
            available_cards = [
                pile.cards[0] for pile in game.supply.piles if len(pile.cards)>0
            ]
            if (village in player.hand.cards):
                yield village
            elif (market in player.hand.cards):
                yield market
            elif (merchant in player.hand.cards):
                yield merchant
            elif (cellar in player.hand.cards) and (
                (
                    estate in player.hand.cards
                ) or (
                    duchy in player.hand.cards
                ) or (
                    province in player.hand.cards
                ) or (
                    curse in player.hand.cards
                )
            ): # The bot will discard victory cards which can't be used
                yield cellar
            elif (mine in player.hand.cards) and (
                (
                    (
                        copper in player.hand.cards
                    ) and (
                        silver in available_cards
                    )
                ) or (
                    (
                        silver in player.hand.cards
                    ) and (
                        gold in available_cards
                    )
                )
            ): # The bot will upgrade coppers to silvers and silvers to golds
                yield mine
            elif (militia in player.hand.cards):
                yield militia
            elif (smithy in player.hand.cards):
                yield smithy
            # elif remodel in player.hand.cards: # Too complex, not used
            #     yield remodel
            elif (workshop in player.hand.cards) and (
                (
                    smithy in available_cards
                ) or (
                    village in available_cards
                ) or (
                    silver in available_cards
                )
            ): # Only get valuable cards, i.e. smithies, villages or silvers
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
            elif len(game.supply.piles[i].cards)==0:
                valid_actions[i] = 0

        # Convert game state to tensor
        state_tensor = torch.tensor(
            self.state,
            dtype=torch.float32
        ).unsqueeze(0)

        # Get valid action mask (1 for valid, 0 for invalid)
        valid_action_mask = torch.tensor([valid_actions])

        # Obtain a decision by the network
        action, _ = player.net.get_action(state_tensor, player.g, valid_action_mask)

        # Set and save the selected action
        self.action = action
        player.memory.states.append(self.state)
        player.memory.actions.append(self.action)

        # Set return card
        card = self.action_mapping[self.action+1]

        if not card:
            return iter([])
        else:
            return iter([card])
