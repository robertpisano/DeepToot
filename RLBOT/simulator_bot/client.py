# sendMessage.py
import socket
import sys
import traceback
from DeepToot.RLBOT.simulator_bot.socket_test import TestSocketClass
from DeepToot.RLBOT.simulator_bot.socket_types import ScenarioData
import pickle
from DeepToot.RLBOT.simulator_bot.msg_protocol import *


#----------------------------------------------------------------------
def sendSocketMessage(class_to_send):
    """
    Send a message to a socket
    """
    try:
        client = socket.socket(socket.AF_INET,
                               socket.SOCK_STREAM)
        client.connect(ADDR)
        client.send(pickle.dumps(class_to_send))
        client.shutdown(socket.SHUT_RDWR)
        client.close()
    except Exception as e:
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    s = ScenarioData()
    s.ball_initial_state.physics.location.x = 500
    s.add_initialization_process(s.initialize_game_state.__name__)
    sendSocketMessage(s)