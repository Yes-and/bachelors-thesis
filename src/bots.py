from typing import Iterator
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
        self.state_before_action = None
        self.state_after_action = None

    def take_turn(self, game: "Game", is_extra_turn: bool = False) -> None:
        self.start_turn(game, is_extra_turn)
        self.start_action_phase(game)
        self.start_treasure_phase(game)
        self.start_buy_phase(game)

        # Save game states before/after action
        self.state_before_action = self.decider.last_state
        self.decider.set_current_state(
            player=self,
            game=game
        )
        self.state_after_action = self.decider.last_state
        self.decider.last_state = None

        self.start_cleanup_phase(game)
        self.end_turn(game)

class MyBotDecider(OptimizedBotDecider):
    def __init__(
        self,
        net,
        epsilon
    ):
        super().__init__()
        self.net = net
        self.epsilon = epsilon
        self.last_state = []
        self.last_action = None
        self.last_reward = None
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
        # Game state
        pile_cards = {str(pile.name): len(pile) for pile in game.supply.piles}

        # Player state
        player_money = player.state.money
        # player_buys = player.state.buys # TODO: Implement multi-buy functionality
        discard_pile_size = len(player.discard_pile)
        player_cards = {name: 0 for name in pile_cards.keys()}
        for card in player.get_all_cards():
            player_cards[str(card)] += 1

        # Opponent state
        i = [str(player) for player in game.players].index('big_money')
        enemy_money = game.players[i].get_deck_money()
        enemy_VPs = game.players[i].get_victory_points()

        # Totals for division (so that state values are normalized)
        pile_cards_div = list(pile_cards.values())
        player_money_div = [216] # 46*1 + 40*2 + 30*3
        discard_pile_div = [250]
        player_cards_div = pile_cards_div
        enemy_money_div = player_money_div
        enemy_VPs_div = [80]
        divs = pile_cards_div + \
            player_money_div + \
            discard_pile_div + \
            player_cards_div + \
            enemy_money_div + \
            enemy_VPs_div
        
        complete_state = list(pile_cards.values()) + \
            [player_money, discard_pile_size] + \
            list(player_cards.values()) + \
            [enemy_money, enemy_VPs]
        normalized_state = np.array(complete_state) / np.array(divs)
        self.last_state = normalized_state
        
        state_indexes = ['pile_' + val for val in list(pile_cards.keys())] + \
            ['player_money', 'player_discard_size'] + \
            ['player_' + val for val in list(player_cards.keys())] + \
            ['enemy_money', 'enemy_VPs']
        if not self.state_indexes:
            self.state_indexes = state_indexes
        else:
            if not (self.state_indexes==state_indexes):
                raise Exception('The order of cards in the state has changed!')

    def set_turn_reward(self, player: "Player", game: "Game"):
        
        VP_delta = 0
        deck_quality = 0
        economic_growth = 0

        # This tracks changes in Victory Points
        card = self.action_mapping[self.last_action]
        if card in [estate, duchy, province]:
            VP_delta = np.tanh(card.score(player) / 5)
        
        # This tracks the proportion of low value cards in deck
        player_cards = [card for card in player.get_all_cards()]
        low_value_cards = [card for card in player_cards if card in [copper, estate, curse]]
        deck_size = len(player_cards)
        deck_quality = 1 - (len(low_value_cards) / deck_size)

        # This tracks increase in purchasing power
        if card in [copper, silver, gold, militia, market]:
            economic_growth = np.tanh(card.money / 2)

        total_reward = 0.4 * VP_delta + 0.3 * deck_quality + 0.3 * economic_growth
        self.last_reward = total_reward

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
        # Automatically set the game state as class variable
        self.set_current_state(player=player, game=game)

        # Basically a function for playing a step
        if np.random.random() < self.epsilon:
            action_indexes = []
            for i in range(1, 18):
                if self.action_mapping[i].base_cost.money <= player.state.money:
                    action_indexes.append(i)
            action = random.choice(action_indexes)

        else:
            state_tensor = torch.tensor(
                self.last_state,
                dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(state_tensor)
                # Action masking (removing unavailable buy options)
                game_supply = {pile.name: len(pile) for pile in game.supply.piles}
                for i in range(1, 18):
                    if (
                        self.action_mapping[i].base_cost.money > player.state.money
                    ) or (
                        game_supply[
                            self.action_mapping[i].name
                        ] == 0
                    ):
                        q_values[0][i-1] = float('-inf')
            action = torch.argmax(q_values).item() + 1

        # Save the action and return the card
        self.last_action = action
        card = self.action_mapping[action]

        # Set continuous reward
        self.set_turn_reward(player, game)

        if not card:
            return iter([])
        else:
            return iter([card])
