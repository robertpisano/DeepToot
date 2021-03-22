from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import DeepToot.RLBOT.simulator_bot.GUI_development.socket_types
import os
from DeepToot.src.managers.SystemManager import SystemManager
from DeepToot.src.comms_util.comms_protocol import CommsProtocol
import traceback

class SocketTestBot(BaseAgent, SystemManager):
    
    def initialize_agent(self):
        SystemManager.__init__(self, index = self.index, ip = CommsProtocol.SERVER, port = CommsProtocol.PORT)
        pass
    
    def get_output(self, game_tick_packet):
        try:
            controllerstate = self.run_bot(packet = game_tick_packet, rigid_packet=self.get_rigid_body_tick(), bot=self)
        except Exception as e:
            controllerstate = SimpleControllerState()
            traceback.print_exc()

        return controllerstate