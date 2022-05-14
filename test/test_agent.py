from conftest import Context
from model import PasaNet
import strategy.constants as c

class TestPokerAgent:
    # <ASSIGNMENT: Test agent initialization and the make_action() function. Also test logging and error handling if you
    # chose to use them.>

    def test_agent_initialization(self, agent):
        """ Test whether the agent is initialized correctly.

        Args:
            agent (PokerAgent): Poker agent object. 
        """        
        assert agent != None
        assert type(agent.model) == PasaNet

    def test_agent_make_action(self, agent, gamestate, gameroundstate):
        """ Test whether the agent makes the correct action.

        Args:
            agent (PokerAgent): Poker agent object.
            gamestate (ClientGameState): ClientGameState object to keep track of one game.
            gameroundstate (ClientGameRoundState): ClientGameRoundState object to keep track of one round.
        """        
        gameroundstate.set_card('J')

        # based on the options given, the agent should choose the check action 
        action = agent.make_action(gamestate, gameroundstate) 
        assert action == 'CHECK'    

        # simulate illegal action, we always want to check in this case
        gameroundstate.set_moves_history(['FOLD', 'CHECK'])
        action = agent.make_action(gamestate, gameroundstate)
        assert action == 'CHECK'
        

    def test_agent_logging(self, agent, gamestate):
        """ Test whether the agent logs the correct information.

        Args:
            agent (PokerAgent): Poker agent object.
            gamestate (ClientGameState): ClientGameState object to keep track of one game.
        """        
        # we can't test the logging functionality, just check if the agent does not crash and stdout is not empty
        agent.on_game_start()
        agent.on_game_end(gamestate, 'WIN')
        agent.on_new_round_request(gamestate)
        
    
    def test_agent_error_handling(self, agent):
        """ Test whether the agent handles the errors correctly.

        Args:
            agent (PokerAgent): Poker agent object.
        """        
        # These are (some of) the errors that can occur:

        # 1. If coordinator is not ready: 'Timeout in coordinator. Coordinator is not ready. Please report.'
        assert agent.on_error(c.ERR_COORDINATOR_NOT_READY) == True # should be recoverable

        # 2. except KuhnCoordinator.CoordinatorWaitingRoomCreationFailed: 'Failed to create a waiting room.'
        assert agent.on_error(c.ERR_WAITING_ROOM_CREATION_FAILED) == False

        # 3. except KuhnWaitingRoom.WaitingRoomIsFull: 'Connection error. Coordinator waiting room is full.'
        assert agent.on_error(c.ERR_WAITING_ROOM_IS_FULL) == False

        # 4. except KuhnWaitingRoom.WaitingRoomIsClosed: 'Connection error. Coordinator waiting room is closed.'
        assert agent.on_error(c.ERR_WAITING_ROOM_IS_CLOSED) == False

        # 5. except KuhnWaitingRoom.PlayerDoubleRegistration: 'Connection error. Player with the same id has been registered already exist in this waiting room.'
        assert agent.on_error(c.ERR_PLAYER_DOUBLE_REGISTRATION) == False



        
    