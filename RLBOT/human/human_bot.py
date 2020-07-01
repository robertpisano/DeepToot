from rlbot.agents.base_agent import BaseAgent
from controller_input import controller
from .RLBOT.srcNeuralNetworkDataGenerator import NeuralNetworkManager, DataGeneratorManager

class Agent(BaseAgent):
    def initialize_agent(self):
        # This runs once before the bot starts up
        # self.controller_state = SimpleControllerState()
        # self.nn_manager = NeuralNetworkManager()
        # self.sc = ScenarioCreator()
        # self.sc.hardcoded_load()

        self.dataManager = DataGeneratorManager(120) # Send in how long we want to save data for 120 frames seems reasonable for testing

    def get_output(self, game_tick_packet):
        if(self.dataManager.started and not self.dataManager.finished): 
            self.dataManager.append(game_tick_packet) # Append data, and will check if complete

        return controller.get_output()
