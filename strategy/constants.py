# This is the constant info defining all cases and their corresponding rewards in a round.
# Reference <https://github.com/int8/counterfactual-regret-minimization/blob/master/common/constants.py>

# All possible cases in a round, e.g. KQ means that player A receives "K" and player B receives "Q" in this round.
import itertools

# Card definitions. The variable 'cards' should be manually updated to match the actual cards in the game.
# In order of LOW to HIGH value:
cards = ['J', 'Q', 'K', 'A'] 
value = {card:idx+1 for idx, card in enumerate(cards)}

# Create an exhaustive list of permutations called CARD_DEALINGS where every entry is a string, a card can't be drawn twice.
CARDS_DEALINGS = list(''.join(x) for x in itertools.permutations(cards, 2))

# Create a dictionary of all possible cases and their corresponding rewards for each entry in CARD_DEALINGS
RESULTS_MAP = {key:1 if value[key[0]] > value[key[1]] else -1 for key in CARDS_DEALINGS}

# Game status: initial cards dealing, i.e. the first action in a round is not limited by the following action rules.
CHANCE = "CHANCE"
# Actions
CHECK = "CHECK"
CALL = "CALL"
FOLD = "FOLD"
BET = "BET"

# The current role, Player A or B.
A = 1
B = -A

# The error cases received from server in the game.
ERR_COORDINATOR_NOT_READY = 'Timeout in coordinator. Coordinator is not ready. Please report.'
ERR_COORDINATOR_CLOSED = 'coordinator.error'
ERR_WAITING_ROOM_CREATION_FAILED = 'Failed to create a waiting room.'
ERR_WAITING_ROOM_IS_FULL = 'Connection error. Coordinator waiting room is full.'
ERR_WAITING_ROOM_IS_CLOSED = 'Connection error. Coordinator waiting room is closed.'
ERR_PLAYER_DOUBLE_REGISTRATION = 'Connection error. Player with the same id has been registered already exist in this waiting room.'
