import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.orientation import Orientation
from util.vec import Vec3

from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState

import sys
import os
sys.path.append(os.path.abspath("D:\Documents\DeepToot\RLBOT\src"))
from NeuralNetworkDataGenerator import NeuralNetworkManager
from ScenarioInterface import ScenarioCreator

class MyBot(BaseAgent):

    def initialize_agent(self):
        # This runs once before the bot starts up
        self.controller_state = SimpleControllerState()
        self.nn_manager = NeuralNetworkManager()
        self.sc = ScenarioCreator()
        self.sc.hardcoded_load()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # self.c = self.c+1
        # print('counter ' + str(self.c))
        # self.nn_manager.get_control_update(packet)
        # print(len(self.nn_manager.memory.get_memory()))
        
        self.sc.control_environment(self)
        print(self.sc.counter)
        return self.controller_state

def set_game_state_from_scenario(self):
    game_state = GameState()
    bpos, bvel = self.sc.get_ball
    ball = BallState(Physics(location=Vec3(bpos), velocity=Vec3(bvel)))
    return GameState(ball=ball)

def find_correction(current: Vec3, ideal: Vec3) -> float:
    # Finds the angle from current to ideal vector in the xy-plane. Angle will be between -pi and +pi.

    # The in-game axes are left handed, so use -x
    current_in_radians = math.atan2(current.y, -current.x)
    ideal_in_radians = math.atan2(ideal.y, -ideal.x)

    diff = ideal_in_radians - current_in_radians

    # Make sure that diff is between -pi and +pi.
    if abs(diff) > math.pi:
        if diff < 0:
            diff += 2 * math.pi
        else:
            diff -= 2 * math.pi

    return diff


def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()
