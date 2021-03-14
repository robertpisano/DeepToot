# sendMessage.py
import socket
import sys
import traceback
from DeepToot.RLBOT.simulator_bot.GUI_development.socket_test import TestSocketClass
from DeepToot.RLBOT.simulator_bot.GUI_development.socket_types import ScenarioData
import pickle
import dill
import yaml
from DeepToot.RLBOT.simulator_bot.GUI_development.msg_protocol import *
from DeepToot.src.meta_data_objects.SerializationFactory import SerializationFactory

#----------------------------------------------------------------------
def sendSocketMessage(class_to_send):
    """
    Send a message to a socket
    """

    try:
        client = socket.socket(socket.AF_INET,
                               socket.SOCK_STREAM)
        client.connect(ADDR)
        # client.send(yaml.dump(class_to_send).encode('utf-8'))
        client.send(SerializationFactory.listify(class_to_send))
        # print(SerializationFactory.listify(class_to_send))
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