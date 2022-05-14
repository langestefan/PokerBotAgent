# Constructing the game tree containing all possible actions and their rewards for all cards-dealing in a round
# Reference <https://github.com/int8/counterfactual-regret-minimization/blob/master/games/kuhn.py>

from strategy.constants import CHECK, BET, CALL, FOLD, A, CHANCE, RESULTS_MAP

class GameStateBase:
    """
    GameStateBase defines the basic info of a node on the game tree.

    Attributes
    ----------
    parent : str
        The parent node.
    to_move : int or str
        It tracks whose turn it is in the current node, whether it is player A, player B, or just a chance node.
        A = 1: player A
        B = -A: player B
        CHANCE: initial cards dealing
    actions : list of str, where str in subset of ['BET', 'CHECK', 'FOLD', 'CALL']
        The possible actions under the parent node. In the root node, actions would be set as the dealt cards.
        In children nodes, it would be the available actions for the current node.
    """    

    def __init__(self, parent, to_move, actions):
        self.parent = parent
        self.to_move = to_move
        self.actions = actions

    def play(self, action):
        return self.children[action]

    def is_chance(self):
        return self.to_move == CHANCE

    def inf_set(self):
        raise NotImplementedError("Please implement information_set method")

class KuhnRootChanceGameState(GameStateBase):
    """
    KuhnRootChanceGameState constructs the game tree.

    Attributes
    ----------
    children : KuhnPlayerMoveGameState
        The children of the current node. The root is initialised as a GameStateBase node with the cards and CHANCE status in a round.
        e.g. root: (None, "CHANCE", "KJ"). "KJ" means that player A has "K", and Player B has "J".
             The first child node: (A, [], "KJ", [BET, CHECK]), which defines the first possible actions for the player A.
        Each child node contains the info in KuhnPlayerMoveGameState.
        The game tree would be constructed for each dealing case in CARDS_DEALINGS.
    _chance_prob : float
        The probability of reaching each node in the next layer should be equal. i.e. 1/len(self.children)
    """ 
    def __init__(self, actions):
        super().__init__(parent = None, to_move = CHANCE, actions = actions)
        self.children = {
            cards: KuhnPlayerMoveGameState(
                self, A, [],  cards, [BET, CHECK]
            ) for cards in self.actions
        }
        self._chance_prob = 1. / len(self.children)

    def is_terminal(self):
        # Defining if the node is the end of a round, initialised as False.
        return False

    def inf_set(self):
        return "."

    def chance_prob(self):
        return self._chance_prob

class KuhnPlayerMoveGameState(GameStateBase):
    """
    KuhnPlayerMoveGameState constructs each child node in the game tree, containing the info of each move. 

    Attributes
    ----------
    actions_history : list of str
        Previously made actions of both players. Actions in the list alternate between players, i.e., the first element
        is the first action of player A, and the second element is the first action of player B, etc. The last element 
        of actions_history is the last action made by the opponent of the current node. If the child node is the first 
        to move, actions_history will be empty.
    cards : str
        The cards dealt in the current round, e.g. "KJ" means that player A has "K", and player B has "J".
    children: KuhnPlayerMoveGameState
        -to_move: achieve the alternate between players.
        actions_history + [a]: add the taken action in the actions_history. Each possible action would be added to the tree as a branch.
        cards: keep the info of the dealt cards.
        __get_actions_in_next_round: get the available choices for the next action.
    public_card:str 
        Current card in hand for the player of this node, e.g. if cards = KJ and to_move = A, which means that 
        the current node records the moving info of player A, then public_card = K. If to_move = -A, then public_card = J.
    _information_set: str
        A player's information set includes the card in hand as well as the action history.
        E.g. if to_move = A, public_card = K, and actions_history = [BET, CALL], then _information_set = [K.BET.CALL].
        The node in the example indicates that player A has K in hand, and that player A BET first, followed by player
        B CALL before this node. 
    """ 

    def __init__(self, parent, to_move, actions_history, cards, actions):
        super().__init__(parent = parent, to_move = to_move, actions = actions)

        self.actions_history = actions_history
        self.cards = cards
        self.children = {
            a : KuhnPlayerMoveGameState(
                self,
                -to_move,
                self.actions_history + [a],
                cards,
                self.__get_actions_in_next_round(a)
            ) for a in self.actions
        }

        public_card = self.cards[0] if self.to_move == A else self.cards[1]
        self._information_set = ".{0}.{1}".format(public_card, ".".join(self.actions_history))

    def __get_actions_in_next_round(self, a):
        """
        Get the available actions this turn, e.g., on the first move, _available_actions = ['BET', 'CHECK', 'FOLD'].

        Parameters
        ----------
        a : the taken action.

        Returns
        -------
        list of str, a list of available actions for the next action
        """
        if len(self.actions_history) == 0 and a == BET:
            return [FOLD, CALL]
        elif len(self.actions_history) == 0 and a == CHECK:
            return [BET, CHECK]
        elif self.actions_history[-1] == CHECK and a == BET:
            return [CALL, FOLD]
        elif a == CALL or a == FOLD or (self.actions_history[-1] == CHECK and a == CHECK):
            return []

    def inf_set(self):
        return self._information_set

    def is_terminal(self):
        """
        Check if the node is at the bottom of the game tree, which means that this node is the end of a round.

        Returns
        -------
        True: there is no available option for the next action.
        False: there are still options for the next action. 
        """
        return self.actions == []

    def evaluation(self):
        """The score of a round is returned if the round ends up."""
        if self.is_terminal() == False:
            raise RuntimeError("trying to evaluate non-terminal node")

        if self.actions_history[-1] == CHECK and self.actions_history[-2] == CHECK:
            return RESULTS_MAP[self.cards] * 1 # only ante is won/lost

        if self.actions_history[-2] == BET and self.actions_history[-1] == CALL:
            return RESULTS_MAP[self.cards] * 2

        if self.actions_history[-2] == BET and self.actions_history[-1] == FOLD:
            return self.to_move * 1
