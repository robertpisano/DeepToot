from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import DeepToot.RLBOT.simulator_bot.GUI_development.socket_types
import os
from DeepToot.RLBOT.simulator_bot.GUI_development.managers.SystemManager import SystemManager
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

class SocketTestBot(BaseAgent):
    # manager = SystemManager(index = 0, ip = '127.0.0.1', port = 5050)
    
    def initialize_agent(self):
        self.manager = SystemManager(index = self.index, ip = '127.0.0.1', port = 5050)
        pass
    
    def get_output(self, game_tick_packet):
        try:
            controllerstate = self.manager.run_bot(RigidBodyTick())
        except:
            # Reset manager
            # self.manager = SystemManager(index = self.index, ip = '127.0.0.1', port = 5050)
            controllerstate = SimpleControllerState()

        return controllerstate


bot = SocketTestBot('test', 'blue', 0)
bot.initialize_agent()

while(True):
    bot.get_output(GameTickPacket())