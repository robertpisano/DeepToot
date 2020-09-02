from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import DeepToot.RLBOT.simulator_bot.socket_types
import os
from DeepToot.RLBOT.simulator_bot.server import ICPThread

class SocketTestBot(BaseAgent):

    def initialize_agent(self):
        self.thread = ICPThread('127.0.0.1', 5050, self)
        self.thread.run()


    def get_output(self, game_tick_packet):
        return SimpleControllerState()