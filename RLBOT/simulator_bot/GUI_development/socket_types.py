from rlbot.agents.base_agent import BaseAgent
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import GameState, GameInfoState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import pickle

# Define classes that will be sent back and forth over the socket
# Initialize ScenarioData() class from data in the GUI, send over socket to bot
# Bot receives this class and sets up game environment

# Send this before sending class over, so the bot knows what object it
# is receiving.

# List of 'keys' to send to server before sending serialized class to prep server?
# Might be unused atm
SCENARIO_DATA_TYPE = 'scenario_data'.encode()


class ScenarioData():

    def __init__(self):
        """Hardcoded initialization. This process will be handled by the GUI interface, currently 
        this function is setup solely for testing.
        """        
        self.scenario_initialization_process_list = []
        self.controller_type = 'empty'
        self.bot = BaseAgent(None, None, None)
        self.game_state = GameState()
        self.game_info_state = GameInfoState(world_gravity_z=-650)
        self.car_initial_state = CarState(physics = Physics(location = Vector3(100, 100, 1000)))
        self.ball_initial_state = BallState(physics = Physics(location = Vector3(400, 100, 1000+92.75), velocity= Vector3(0,0,0), rotation=Rotator(0,0,0), angular_velocity=Vector3(0.0001,0,0)))

    def add_initialization_process(self, method_name):
        """[Add the name of the method (from this class)
         to add to the list of methods that will run when this class is received by the server.
         This is used to set the Rocket League game environment before a scenario.]

        Args:
            method_name ([type]): [description]
        """        
        self.scenario_initialization_process_list.append(method_name)
    
    def initialize_game_info_state(self):
        """Initialize the environment parameters. Mainly gravity
        """        
        self.bot.set_game_state(GameState(game_info = self.game_info_state))

    def initialize_game_state(self):
        """Initialize state of ball and cars called the game state
        """        
        self.game_state.ball = self.ball_initial_state
        self.game_state.cars = {self.bot.index: self.car_initial_state}
        self.bot.set_game_state(self.game_state)

    def return_serialized(self):
        """Returns serialized version of self

        Returns:
            [type]: [description]
        """        
        return pickle.dumps(self)