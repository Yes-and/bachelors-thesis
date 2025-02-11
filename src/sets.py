from pyminion.expansions.base import (
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
from pyminion.core import Card

# Cards selected according to rulebook: https://wiki.dominionstrategy.com/images/c/c5/DominionRulebook2016.pdf
custom_set: list[Card] = [
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
]