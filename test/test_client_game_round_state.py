from conftest import Context
from client.state import ClientGameRoundState

class TestClientGameRoundState:

    # <ASSIGNMENT: Test the creation, reading and updating of ClientGameRoundState>

    def test_game_round_state_creation(self, gameroundstate):
        """ Create a gameroundstate object and test if it actually exists.
        
        Args:
            gameroundstate (ClientGameRoundState): ClientGameRoundState object to keep track of one round.
        """ 
        assert gameroundstate != None
        assert type(gameroundstate) == ClientGameRoundState

    def test_reading_game_round_state(self, gameroundstate):    
        """ Test reading the gameroundstate options

        Args:
            gameroundstate (ClientGameRoundState): ClientGameRoundState object to keep track of one round.
        """
        assert gameroundstate.get_coordinator_id() == Context.coordinator_id
        assert gameroundstate.get_round_id() == 1  # game just started, round_id should be 1
        assert gameroundstate.get_available_actions() == Context.available_actions
        assert gameroundstate.get_cards() == Context.cards
        assert gameroundstate.get_moves_history() == Context.moves_history

    def test_updating_game_round_state(self, gameroundstate):
        """ Test updating the gameroundstate options  

        Args:
            gameroundstate (ClientGameRoundState): ClientGameRoundState object to keep track of one round.
        """ 
        # update the moves history
        updated_history = Context.moves_history.append('BET')
        gameroundstate.set_moves_history(updated_history)
        assert gameroundstate.get_moves_history() == updated_history

        # update the available actions
        updated_actions = gameroundstate.get_available_actions()
        updated_actions.append('BET')
        assert gameroundstate.get_available_actions() == updated_actions

        # update the outcome
        outcome = '-4'
        assert gameroundstate.get_outcome() == None # check if outcome at start of round is empty
        gameroundstate.set_outcome(outcome)  # round has happened, set new outcome          
        assert gameroundstate.get_outcome() == outcome  # check if outcome was updated

            

