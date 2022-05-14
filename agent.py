import os
import json
from pasanet_nn import PasaNet
import strategy.constants as c
from model import load_model, identify
from client.state import ClientGameRoundState, ClientGameState
from strategy.agent_strategy import create_strategy
from time import sleep, time


class PokerAgent(object):

    def __init__(self):
        self.model = load_model(PasaNet(), r'PasaNet')

    def make_action(self, state: ClientGameState, round: ClientGameRoundState) -> str:
        """
        Next action, used to choose a new action depending on the current state of the game. This method implements your
        unique PokerBot strategy. Use the state and round arguments to decide your next best move.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game (a game has multiple rounds)
        round : ClientGameRoundState
            State object of the current round (from deal to showdown)

        Returns
        -------
        str in ['BET', 'CALL', 'CHECK', 'FOLD'] (and in round.get_available_actions())
            A string representation of the next action an agent wants to do next, should be from a list of available actions
        """

        # <ASSIGNMENT: Replace this random strategy with your own implementation and update file(s) accordingly. Make
        # your implementation robust against the introduction of additional players and/or cards.>
        
        # create the strategy dictionary if it does not exist
        if not os.path.exists("agent_strategy.json"):
            create_strategy()
        # load strategy
        tf = open("agent_strategy.json", "r")
        strategy = json.load(tf)
        # get info_set of the current round as the key
        card = round.get_card()
        actions_history = round.get_moves_history()
        info_key = ".{0}.{1}".format(card, ".".join(actions_history))
        # find the action with the highest utility corresponding to the key
        try:
            action = max(strategy[info_key], key = strategy[info_key].get)
        except KeyError:
            print("Caught Exception KeyError:", info_key)
            print("Checking due to KeyError")
            action = "CHECK"
        return action

    def on_image(self, image):
        """
        This method is called every time when card image changes. Use this method for image recongition procedure.

        Parameters
        ----------
        image : Image
            Image to classify.
        
        Returns
        -------
        rank : str in ['J', 'Q', 'K', 'A'] 
        Estimated card rank.
        """
        # identify the card and return the rank
        return identify(image, self.model)
        

    def on_error(self, error):
        """
        This method will be called in case of error either from server backend or from client itself. You can
        optionally use this function for error handling.

        Parameters
        ----------
        error : str
            string representation of the error
        """
        # we can't recover from this error, so we just print it
        unrecoverable = {c.ERR_WAITING_ROOM_CREATION_FAILED, 
                        c.ERR_WAITING_ROOM_IS_FULL, 
                        c.ERR_WAITING_ROOM_IS_CLOSED, 
                        c.ERR_PLAYER_DOUBLE_REGISTRATION}

        recoverable = {c.ERR_COORDINATOR_NOT_READY}

        # if error is unrecoverable, close the client
        if error is not None and error in unrecoverable:
            print('Unrecoverable error, please restart the client. Error: ', error)
            return False

        # if error is recoverable, wait continue the game
        if error is not None and error in recoverable:
            print('Recoverable error, please continue the game. Error: ', error)
            print('Waiting for 1 second...')
            sleep(1)
            return True
            

    def on_game_start(self):
        """
        This method will be called once at the beginning of the game when server confirms both players have connected.
        """
        print('Game started!')

    def on_new_round_request(self, state: ClientGameState):
        """
        This method is called every time before a new round is started. A new round is started automatically.
        You can optionally use this method for logging purposes.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        """
        print("New round. Bank:", state.get_player_bank())

    def on_round_end(self, state: ClientGameState, round: ClientGameRoundState):
        """
        This method is called every time a round has ended. A round ends automatically. You can optionally use this
        method for logging purposes.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        round : ClientGameRoundState
            State object of the current round
        """
        print(round.get_round_id(), f'[{round.get_card()}|{round.get_turn_order()}]', ':', round.get_moves_history(), '->',
              f'{round.get_outcome()}|{round.get_cards()}')

    def on_game_end(self, state: ClientGameState, result: str):
        """
        This method is called once after the game has ended. A game ends automatically. You can optionally use this
        method for logging purposes.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        result : str in ['WIN', 'DEFEAT']
            End result of the game
        """
        print(f'Result: {result}. Bank: {state.get_player_bank()}\n')

