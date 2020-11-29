from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import DeepToot.RLBOT.simulator_bot.GUI_development.socket_types
import os
from DeepToot.RLBOT.simulator_bot.GUI_development.managers.SystemManager import SystemManager

class SocketTestBot(BaseAgent):
    # manager = SystemManager(index = 0, ip = '127.0.0.1', port = 5050)
    
    def initialize_agent(self):
        self.manager = SystemManager(index = self.index, ip = '127.0.0.1', port = 5050)
        pass
    
    def get_output(self, game_tick_packet):
        controllerstate = self.manager.run_bot(self.get_rigid_body_tick())
        return controllerstate