# Tests for the Kuhn game tree and cfr calculation
# Reference <https://github.com/int8/counterfactual-regret-minimization/blob/master/tests/test_kuhn.py>

from strategy.kuhn_state_tree import KuhnRootChanceGameState
from strategy.crf_algorithm import VanillaCFR
from strategy.constants import *
from strategy.agent_strategy import create_strategy
import os
import pytest

def __recursive_tree_assert(root, logical_expression):
    """ 
    The nodes of a tree need a recursive search

    Args:
        root:  KuhnRootChanceGameState, the root of the game tree.
        logical_expression: assertation statement for a node.
    """ 
    assert logical_expression(root)
    for k in root.children:
        __recursive_tree_assert(root.children[k], logical_expression)

def test_kuhn_tree_actions_number_equal_to_children():
    # generate the game tree 
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    # test recursively
    __recursive_tree_assert(root, lambda node: len(node.children) == len(node.actions))

def test_kuhn_to_move_chance_at_root():
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    assert root.to_move == CHANCE

def test_kuhn_to_move_changes_correctly_for_children():
    logical_expression = lambda node: all([node.to_move == -node.children[k].to_move for k in node.children])
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    for k in root.children:
        child = root.children[k]
        __recursive_tree_assert(child, logical_expression)

def test_player_a_acts_first():
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    for k in root.children:
        child = root.children[k]
        assert child.to_move == A

def test_if_only_root_is_chance():
    logical_expression = lambda node: not node.is_chance()
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    assert root.is_chance()
    for k in root.children:
        child = root.children[k]
        __recursive_tree_assert(child, logical_expression)

def test_if_possible_to_play_unavailable_action():
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    with pytest.raises(KeyError):
        root.play(CALL)
    with pytest.raises(KeyError):
        root.play(BET).play(BET)
    with pytest.raises(KeyError):
        root.play(CHECK).play(CALL)

def test_inf_sets():
    """
    Test if the information set of each node on the game tree is correct.
    """
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    for cards in CARDS_DEALINGS:
        assert root.inf_set() == "."
        assert root.play(cards).inf_set() == ".{0}.".format(cards[0])
        assert root.play(cards).play(BET).inf_set() == ".{0}.BET".format(cards[1])
        assert root.play(cards).play(BET).play(CALL).inf_set() == ".{0}.BET.CALL".format(cards[0])
        assert root.play(cards).play(CHECK).play(BET).play(FOLD).inf_set() == ".{0}.CHECK.BET.FOLD".format(cards[1])

def test_termination():
    """
    Test if the end node of a round on the game tree is recognized correctly.
    """
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    assert not root.is_terminal()

    for cards in CARDS_DEALINGS:
        # not terminal node
        assert not root.play(cards).play(BET).is_terminal()
        assert not root.play(cards).play(CHECK).is_terminal()
        assert not root.play(cards).play(CHECK).play(BET).is_terminal()
        assert not root.play(cards).play(BET).is_terminal()
    
        # terminal node
        assert root.play(cards).play(BET).play(FOLD).is_terminal()
        assert root.play(cards).play(CHECK).play(CHECK).is_terminal()
        assert root.play(cards).play(BET).play(CALL).is_terminal()
        assert root.play(cards).play(CHECK).play(BET).play(FOLD).is_terminal()
        assert root.play(cards).play(CHECK).play(BET).play(CALL).is_terminal()
        assert root.play(cards).play(BET).play(CALL).is_terminal()

def test_evaluation():
    """
    Test if the reward of the end node on the game tree is correct.
    """
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    for cards in CARDS_DEALINGS:
        node = root.children[cards]
        if value[cards[0]] > value[cards[1]]:
            assert node.play(BET).play(FOLD).evaluation() == 1
            assert node.play(BET).play(CALL).evaluation() == 2
            assert node.play(CHECK).play(BET).play(FOLD).evaluation() == -1
            assert node.play(CHECK).play(CHECK).evaluation() == 1
        else:
            assert node.play(BET).play(FOLD).evaluation() == 1
            assert node.play(BET).play(CALL).evaluation() == -2
            assert node.play(CHECK).play(BET).play(FOLD).evaluation() == -1
            assert node.play(CHECK).play(CHECK).evaluation() == -1

def test_if_actions_in_strategy_dictionary_valid():
    """
    Test if the keys in the strategy corresponding to inf_set are valid.
    All inf_set should be valid after passing the above tests for the game tree.
    """
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    vanilla_cfr = VanillaCFR(root)
    vanilla_cfr.run(iterations = 1000)
    vanilla_cfr.compute_nash_equilibrium()
    
    inf_set = ".J.BET.CALL"
    assert vanilla_cfr.nash_equilibrium[inf_set] == {}

    inf_set = ".Q.BET"
    valid_action = "FOLD"
    assert valid_action in vanilla_cfr.nash_equilibrium[inf_set]

    inf_set = ".A.CHECK.BET"
    invalid_action = "CHECK"
    assert invalid_action not in vanilla_cfr.nash_equilibrium[inf_set]

def test_no_negative_utility_in_strategy():
    """
    Test if the utility of each action in the strategy is valid (no negative values)
    """
    root = KuhnRootChanceGameState(CARDS_DEALINGS)
    vanilla_cfr = VanillaCFR(root)
    vanilla_cfr.run(iterations = 1000)
    vanilla_cfr.compute_nash_equilibrium()
    
    for item in vanilla_cfr.nash_equilibrium.values():
        for utility in item.values():
            assert utility >= 0 

def test_create_strategy_file():
    """
    Test if the required strategy file is generated
    """
    if os.path.exists("agent_strategy.json"):
        os.remove("agent_strategy.json")
    
    create_strategy()
    assert os.path.exists("agent_strategy.json")

