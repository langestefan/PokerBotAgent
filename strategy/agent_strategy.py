from strategy.constants import CARDS_DEALINGS
from strategy.kuhn_state_tree import KuhnRootChanceGameState
from strategy.crf_algorithm import VanillaCFR
import json

def create_strategy():
    """create the strategy dictionary based on crf computation"""

    #Construct the game tree based on the CARDS_DEALINGS(4 cards)
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    #Compute the utility of nodes for 1000 iterations to approximate the nash equilibrium
    vanilla_cfr = VanillaCFR(root)
    vanilla_cfr.run(iterations = 1000)
    vanilla_cfr.compute_nash_equilibrium()
    #save the strategy dictionary in a .json file
    tf = open("agent_strategy.json", "w")
    json.dump(vanilla_cfr.nash_equilibrium,tf)
    tf.close()