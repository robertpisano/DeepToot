from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import DeepToot.RLBOT.simulator_bot.GUI_development.socket_types
import os
from DeepToot.src.managers.SystemManager import SystemManager
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlbot.utils.structures.rigid_body_struct import RigidBodyTick
from DeepToot.src.comms_util.comms_protocol import CommsProtocol
import traceback
class SocketTestBot(BaseAgent, SystemManager):
    # manager = SystemManager(index = 0, ip = '127.0.0.1', port = 5050)
    
    def initialize_agent(self):
        SystemManager.__init__(self, index = self.index, ip = CommsProtocol.SERVER, port = CommsProtocol.PORT)
        pass
    
    def get_output(self, game_tick_packet):
        try:
            controllerstate = self.run_bot(packet = game_tick_packet, rigid_packet=RigidBodyTick(), bot=self)
        except:
            # Reset manager
            # self.manager = SystemManager(index = self.index, ip = '127.0.0.1', port = 5050)
            traceback.print_exc()
            controllerstate = SimpleControllerState()

        return controllerstate


if __name__ == "__main__":
    bot = SocketTestBot('test', 'blue', 0)
    bot.initialize_agent()
    import time

    while True:
        bot.get_output(GameTickPacket())
        time.sleep(1)