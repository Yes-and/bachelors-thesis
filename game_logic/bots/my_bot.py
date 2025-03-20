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
        player_id: str = "my_bot"
    ):
        super().__init__(
            decider=MyBotDecider(
                net
            ),
            player_id=player_id
            )
        self.state_before_action = None
        self.state_after_action = None

    def take_turn(self, game: "Game", is_extra_turn: bool = False) -> None:
        self.start_turn(game, is_extra_turn)
        self.start_action_phase(game)
        self.start_treasure_phase(game)
        
        # Save game states before action
        self.decider.set_current_state(self, game)
        self.state_before_action = self.decider.state
        self.start_buy_phase(game)
        self.decider.reset_state() # Reset the state for debugging

        self.start_cleanup_phase(game)
        self.end_turn(game)

class MyBotDecider(OptimizedBotDecider):
    def __init__(
        self,
        net
    ):
        super().__init__()
        self.net = net
        self.state = None
        self.action = None
        self.state_value = None
        self.reward = None
        self.log_prob = None
        self.done = None
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

        # Player state
        # player_money = player.get_deck_money()
        # player_vp = player.get_victory_points()

        # Opponent state
        # i = [str(player) for player in game.players].index('big_money')
        # enemy_money = game.players[i].get_deck_money()
        # enemy_vp = game.players[i].get_victory_points()

        # Totals for division (so that state values are normalized)
        pile_cards_div = [8, 8, 8, 10, 46, 40, 30, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        player_cards_div = pile_cards_div

        player_money_div = [216] # 46*1 + 40*2 + 30*3
        player_vp_div = [80]
        enemy_money_div = player_money_div
        enemy_vp_div = [80]

        divs = pile_cards_div + \
            player_cards_div
        complete_state = list(pile_cards.values()) + \
            list(player_cards.values())
        normalized_state = np.array(complete_state) / np.array(divs)
        if not sum(normalized_state) > 0:
            pass
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
        # while (player.state.actions > 0):
        #     if village in player.hand.cards:
        #         yield village
        #     elif market in player.hand.cards:
        #         yield market
        #     elif merchant in player.hand.cards:
        #         yield merchant
        #     elif cellar in player.hand.cards:
        #         yield cellar
        #     elif mine in player.hand.cards:
        #         yield mine
        #     elif militia in player.hand.cards:
        #         yield militia
        #     elif smithy in player.hand.cards:
        #         yield smithy
        #     elif remodel in player.hand.cards:
        #         yield remodel
        #     elif workshop in player.hand.cards:
        #         yield workshop
        #     elif moat in player.hand.cards:
        #         yield moat
        #     else:
                # return iter([])
        return iter([])

    @torch.no_grad()
    def buy_priority(self, player: "Player", game: "Game") -> Iterator[Card]:
        # Automatically set the game state as class variable
        # self.set_current_state(player=player, game=game)
        state_tensor = torch.tensor(
            self.state,
            dtype=torch.float32
        ).unsqueeze(0)

        # Get logits (before softmax) and state value from policy network
        action_logits, state_value = self.net(state_tensor)  # Logits, not probabilities

        # Get valid actions
        valid_actions = [1] * 18
        cards = list(self.action_mapping.values())[:17]
        for i in range(17):
            if cards[i].base_cost.money > player.state.money:
                valid_actions[i] = 0

        # Get valid action mask (1 for valid, 0 for invalid)
        valid_action_mask = torch.tensor([valid_actions], device=action_logits.device)

        # Mask invalid actions by setting logits of invalid actions to a large negative number (-1e9)
        masked_logits = action_logits + (valid_action_mask - 1) * 1e9  # -1 * 1e9 → Very negative logits

        # Apply softmax to get final probability distribution
        action_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Sample action from the masked probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        log_prob = action_dist.log_prob(torch.tensor(action, device=action_logits.device))

        # Save important information
        self.action = action
        self.state_value = state_value.item()
        self.log_prob = log_prob.detach().numpy()
        self.reward = 0
        self.done = 0

        # Set return card
        card = self.action_mapping[action+1]

        # if card in [duchy, province]:
        #     reward = card.score(player)
        #     self.reward = reward
        # elif card in [silver, gold]:
        #     reward = card.money
        #     self.reward = reward

        if not card:
            return iter([])
        else:
            return iter([card])
