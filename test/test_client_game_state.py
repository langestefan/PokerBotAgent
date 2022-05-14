from conftest import Context
from client.state import ClientGameState

class TestClientGameState:

    # <ASSIGNMENT: Test the creation, reading and updating of ClientGameState>

    def test_game_state_creation(self, gamestate):
        """ Create a gamestate object and test if it actually exists.

        Args:
            gamestate (ClientGameState): ClientGameState object to keep track of one game.
        """        
        assert gamestate != None
        assert type(gamestate) == ClientGameState

    def test_reading_game_state(self, gamestate):    
        """ Test reading the gamestate options

        Args:
            gamestate (ClientGameState): ClientGameState object to keep track of one game.
        """       
        assert gamestate.get_coordinator_id() == Context.coordinator_id
        assert gamestate.get_player_token() == Context.token
        assert gamestate.get_player_bank() == Context.start_bank  # game just started, player bank still full

    def test_updating_game_state(self, gamestate):         
        """ Test updating the gamestate options

        Args:
            gamestate (ClientGameState): ClientGameState object to keep track of one game.
        """           
        old_bank = gamestate.get_player_bank()
        profit = 3

        # hypothetical round has happened, update the bank
        gamestate.update_bank(profit)
        assert gamestate.get_player_bank() == profit + old_bank   

        # start a round
        gamestate.start_new_round()        
        round_id = gamestate.get_last_round_state().get_round_id()
        gamestate.start_new_round() # start a new round.
        assert gamestate.get_last_round_state().get_round_id() == round_id + 1

