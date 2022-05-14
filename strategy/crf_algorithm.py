# Calculating the utility for each node on the game tree based on the Counter factual Regret Minimization algorithm
# Reference <https://github.com/int8/counterfactual-regret-minimization/blob/master/games/algorithms.py>

from strategy.constants import A

def init_sigma(node, output = None):
    """
    The initialisation of the sigma value, which is the expected utility of the next action.
    Sigma values are basically normalised regret values. The stratey value for each child of a node
    on the game tree is initialised equally, i.e. 1 / the number of branch. The whole tree is initialised recursively
    starting from the root.

    Parameters
    ----------
    node : KuhnRootChanceGameState, the root of the generalised game tree.

    Returns
    -------
    output: dict
        The dictionary contains the initialised sigma value of each node on the tree.
        E.g. Node [K.BET]==>the corresponding initialised content in the dictionary:{FOLD : 0.5, CALL : 0.5 }
    """
    output = dict()
    def init_sigma_recursive(node):
        output[node.inf_set()] = {action: 1. / len(node.actions) for action in node.actions}
        for k in node.children:
            init_sigma_recursive(node.children[k])
    init_sigma_recursive(node)
    return output

def init_empty_node_maps(node, output = None):
    """
    Construct the map with empty utility values for the game tree recursively starting from the root. 
    E.g. for the node [K.BET], the keys of this enry should be all the next available actions, i.e.[FOLD: , CALL: ].

    Parameters
    ----------
    node : KuhnRootChanceGameState, the root of the generalised game tree.

    Returns
    -------
    output: dict
        The dictionary with all keys and empty values. 
    """
    output = dict()
    def init_empty_node_maps_recursive(node):
        output[node.inf_set()] = {action: 0. for action in node.actions}
        for k in node.children:
            init_empty_node_maps_recursive(node.children[k])
    init_empty_node_maps_recursive(node)
    return output

class CounterfactualRegretMinimizationBase:
    """
    Defining the functions used for calculating the strategy value of each operation based on the game tree of Kuhn Poker.

    Attributes
    ----------
    root : KuhnRootChanceGameState
        The root of the game tree, see details in KuhnRootChanceGameState.
    sigma : float
        The expected utility of each choice, i.e. normalised regret values.
    cumulative_regrets : dict
        The cumulative regret value of each choice. Regret is a difference between the actual rewards and the hypothetical rewards.
        The higher the regret value of an action, the greater the utility of the action.
        E.g. for the node [K.BET], the player can get 1 point when taking CALL while lose 1 point when taking FOLD, the corresponding
        regret value should be {CALL: 0, FOLD: 2} , which means the player would lose 2 points compared with the best single reward
        if the next action is not FOLD. 
    cumulative_sigma : dict
        The cumulative utility of an info set is calculated based on regret values. Higher regret ==> Higher sigma.
    nash_equilibrium : dict
        The final utility of each choice which can approximate Nash Equilibrium. If two no-regret algorithms(A and B) play a zero-sum game 
        against one another for T iterations with average regret less than ϵ, then their average strategies approximate Nash Equilibrium.
    """

    def __init__(self, root):
        self.root = root
        self.sigma = init_sigma(root)
        self.cumulative_regrets = init_empty_node_maps(root)
        self.cumulative_sigma = init_empty_node_maps(root)
        self.nash_equilibrium = init_empty_node_maps(root)

    def _update_sigma(self, i):
        #normalise the sigma
        rgrt_sum = sum(filter(lambda x : x > 0, self.cumulative_regrets[i].values()))
        for a in self.cumulative_regrets[i]:
            self.sigma[i][a] = max(self.cumulative_regrets[i][a], 0.) / rgrt_sum if rgrt_sum > 0 else 1. / len(self.cumulative_regrets[i].keys())

    def compute_nash_equilibrium(self):
        self.__compute_ne_rec(self.root)

    def __compute_ne_rec(self, node):
        if node.is_terminal():
            return
        i = node.inf_set()
        if node.is_chance():
            self.nash_equilibrium[i] = {a:node.chance_prob() for a in node.actions}
        else:
            sigma_sum = sum(self.cumulative_sigma[i].values())
            self.nash_equilibrium[i] = {a: self.cumulative_sigma[i][a] / sigma_sum for a in node.actions}
        # go to subtrees
        for k in node.children:
            self.__compute_ne_rec(node.children[k])

    def _cumulate_cfr_regret(self, information_set, action, regret):
        self.cumulative_regrets[information_set][action] += regret

    def _cumulate_sigma(self, information_set, action, prob):
        self.cumulative_sigma[information_set][action] += prob

    def run(self, iterations):
        raise NotImplementedError("Please implement run method")

    def _cfr_utility_recursive(self, state, reach_a, reach_b):
        """
        Calculate the regrets value and utility of each actions recursively starting from the root.

        Parameters
        ----------
        state : KuhnRootChanceGameState
            The root of the game tree
        reach_a : float
            The base utility when player A reaches the node, initialised as 1.
        reach_b : float
            The base utility when player B reaches the node, initialised as 1.
            
        Returns
        -------
        value: float
            The utility value of a node.
        """
        children_states_utilities = {}
        if state.is_terminal():
            # evaluate terminal node according to the game result
            return state.evaluation()
        if state.is_chance():
            chance_outcomes = {state.play(action) for action in state.actions}
            return state.chance_prob() * sum([self._cfr_utility_recursive(outcome, reach_a, reach_b) for outcome in chance_outcomes])
        # sum up all utilities for playing actions in our game state
        value = 0.
        for action in state.actions:
            child_reach_a = reach_a * (self.sigma[state.inf_set()][action] if state.to_move == A else 1)
            child_reach_b = reach_b * (self.sigma[state.inf_set()][action] if state.to_move == -A else 1)
            # value as if child state implied by chosen action was a game tree root
            child_state_utility = self._cfr_utility_recursive(state.play(action), child_reach_a, child_reach_b)
            # value computation for current node
            value +=  self.sigma[state.inf_set()][action] * child_state_utility
            # values for chosen actions (child nodes) are kept here
            children_states_utilities[action] = child_state_utility
        # we are computing regrets for both players simultaneously, therefore we need to relate reach,reach_opponent to the player acting
        # in current node, for player A, it is different than for player B
        (cfr_reach, reach) = (reach_b, reach_a) if state.to_move == A else (reach_a, reach_b)
        for action in state.actions:
            # we multiply regret by -1 for player B, this is because value is computed from player A perspective
            # again we need that perspective switch
            action_cfr_regret = state.to_move * cfr_reach * (children_states_utilities[action] - value)
            self._cumulate_cfr_regret(state.inf_set(), action, action_cfr_regret)
            self._cumulate_sigma(state.inf_set(), action, reach * self.sigma[state.inf_set()][action])
        return value


class VanillaCFR(CounterfactualRegretMinimizationBase):

    """VanillaCFR runs the CRF algorithm"""
    
    def __init__(self, root):
        super().__init__(root = root)

    def run(self, iterations = 1):
        for _ in range(0, iterations):
            self._cfr_utility_recursive(self.root, 1, 1)
            # since we do not update sigmas in each information set while traversing, we need to
            # traverse the tree to perform to update it now
            self.__update_sigma_recursively(self.root)

    def __update_sigma_recursively(self, node):
        # stop traversal at terminal node
        if node.is_terminal():
            return
        # omit chance
        if not node.is_chance():
            self._update_sigma(node.inf_set())
        # go to subtrees
        for k in node.children:
            self.__update_sigma_recursively(node.children[k])
