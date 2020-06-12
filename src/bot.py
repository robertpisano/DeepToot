import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# from rlbot.utils.game_state_util import GameState
# from rlbot.utils.game_state_util import CarState
# from rlbot.utils.game_state_util import Physics
# from rlbot.utils.game_state_util import Vector3
# from rlbot.utils.game_state_util import Rotator
# from rlbot.utils.game_state_util import BallState

from DeepToot.src.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder
from DeepToot.src.data_generation.state_transformer import StateTransformer


class MyBot(BaseAgent):
    game_trajectory_builder = None

    def initialize_agent(self):
        self.game_trajectory_builder = GameTrajectoryBuilder(10)
        self.opp_index = self.get_opponent_index()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.game_trajectory_builder.add_game_state(ball_state = StateTransformer.from_game_tick_packet_to_ball_state(packet=packet),
                                                    bot_state = StateTransformer.from_game_tick_packet_to_car_state(packet=packet, index = self.index),
                                                    opp_state = StateTransformer.from_game_tick_packet_to_car_state(packet=packet, index = self.opp_index))

        
        game_trajectory = self.game_trajectory_builder.build()
        self.controller_state = SimpleControllerState() # Set controller state to null state
        print(game_trajectory.BOT_TRAJECTORY.states[-1])
        return self.controller_state

    def get_opponent_index(self):
        """ONLY WORKS FOR 1V1
            based off of bots index, assume opponents index is the opposite between 0 and 1

        Returns:
            [type]: [description]
        """    
        if(self.index == 0):
            return 1
        else:
            return 0

def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()

if __name__ == "__main__":
    None