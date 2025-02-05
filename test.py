import collections
import logging

from lib.my_set import my_set
from lib.my_bot import MyBot

import numpy as np
from pyminion.bots.examples import BigMoney
from pyminion.game import Game



DECISIONS_MAPPING = list()

logger = logging.getLogger()

Experience = collections.namedtuple(
    'Experience', field_names=[
        'state',
        'action',
        'reward',
        'new_state',
        'done'
    ]
)

if __name__ == '__main__':

    bot1 = BigMoney()
    bot2 = MyBot()

    last_experience = Experience(
        state = None,
        action = None,
        reward = None,
        new_state = None,
        done = False
    )

    game = Game(players=[bot1, bot2], expansions=[my_set])
    game.start()

    all_card_names = [val.name for val in game.supply.basic_score_piles] + \
        [val.name for val in game.supply.basic_treasure_piles] + \
        [val.name for val in game.supply.kingdom_piles]

    while True:
        for player in game.players:
            game.current_player = player

            # replace game.play_turn(player)
            extra_turn_count = 0
            take_turn = True
            while take_turn:

                # replace player.take_turn(game, is_extra_turn=extra_turn_count > 0)
                player.start_turn(game, extra_turn_count > 0)
                player.start_action_phase(game)
                player.start_treasure_phase(game)
                
                player_money = player.state.money
                pile_cards = {val.name: len(val) for val in game.supply.basic_score_piles} | \
                    {val.name: len(val) for val in game.supply.basic_treasure_piles} | \
                    {val.name: len(val) for val in game.supply.kingdom_piles}
                player_cards = {name: 0 for name in all_card_names}
                for card in player.get_all_cards():
                    player_cards[str(card)] += 1

                current_state = np.array([player_money] + list(pile_cards.values()) + list(player_cards.values()))
                last_experience = Experience(
                    state = current_state,
                    action = None,
                    reward = None,
                    new_state = None,
                    done = False
                )

                # replace player.start_buy_phase(game)
                while player.state.buys > 0:
                    logger.info(game.supply.get_pretty_string(player, game))
                    logger.info(f"Money: {player.state.money}")
                    if player.state.potions > 0:
                        logger.info(f"Potions: {player.state.potions}")
                    logger.info(f"Buys: {player.state.buys}")

                    valid_cards = [
                        c
                        for c in game.supply.available_cards()
                        if c.get_cost(player, game).money <= player.state.money and
                        c.get_cost(player, game).potions <= player.state.potions
                    ]
                    card = player.decider.buy_phase_decision(
                        valid_cards=valid_cards,
                        player=player,
                        game=game,
                    )
                    if not DECISIONS_MAPPING:
                        DECISIONS_MAPPING = list(pile_cards.keys())+['None']
                    current_action = DECISIONS_MAPPING.index(str(card))
                    current_reward = 0
                    # add case for buying a curse
                    if current_action == 3:
                        current_reward -= 1
                    elif current_action == 4:
                        current_reward -= 1
                    elif current_action == 0:
                        current_reward -= 1
                    
                    # action is equal to which card is bought
                    last_experience = Experience(
                        state = current_state,
                        action = current_action,
                        reward = current_reward,
                        new_state = None,
                        done = False
                    )

                    if card is None:
                        logger.info(f"{player} buys nothing")
                        break

                    player.buy(card, game)
                    # next state can be found here
                    pass

                game.effect_registry.on_buy_phase_end(player, game)
                
                player.start_cleanup_phase(game)
                player.end_turn(game)

                # reset card cost reduction
                game.card_cost_reduction = 0

                if game.is_over():
                    break

                extra_turn_count += 1
                take_turn = player.take_extra_turn and extra_turn_count < 2

        if game.is_over():
            result = game.summarize_game()
            # Complete the experience and add it to the replay buffer
            last_experience = Experience(
                last_experience[0],
                last_experience[1],
                last_experience[2],
                last_experience[3],
                True
            )
            # TODO: add to replay buffer
            print(result)
