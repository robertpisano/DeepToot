from rlbot.utils.structures.game_data_struct import GameTickPacket
from DeepToot.src.data_generation.entities.state.ball_state import BallState
from DeepToot.src.data_generation.entities.state.car_state import CarState
import pandas as pd
import numpy as np
from DeepToot.src.data_generation.entities.physics.base_3d_vector import Base3DVector
from pyquaternion import Quaternion

# StateTransformer will convert "raw data" types (ex: GameTickPacket, pandas.DataFrame) into 
# respective state class for the trajectory builder
class StateTransformer():
    @staticmethod
    def from_game_tick_packet_to_ball_state(packet: GameTickPacket):
        """returns a ball state given a game tick packet

        Args:
            packet (GameTickPacket): the game tick packet from the rlbot library

        Returns:
            BallState: the state that we want to use 
        """        
        return BallState(position = packet.game_ball.physics.location,
                        velocity = packet.game_ball.physics.velocity,
                        ang_vel = packet.game_ball.physics.angular_velocity,
                        orientation = packet.game_ball.physics,
                        time = packet.game_info.seconds_elapsed)

    @staticmethod
    def from_game_tick_packet_to_car_state(packet: GameTickPacket, index: int):
        """gives the car state of our current player

        Args:
            packet (GameTickPacket): the game tick packet from the rlbot library
            index ([integer]): the position of the car's data in the GameTickPacket

        Returns:
            CarState: the state that we want to use
        """        
        return CarState(position = packet.game_cars[index].physics.location,
                velocity = packet.game_cars[index].physics.velocity,
                ang_vel = packet.game_cars[index].physics.angular_velocity,
                orientation = packet.game_cars[index].physics,
                time = packet.game_info.seconds_elapsed,
                hit_box = packet.game_cars[index].hitbox,
                is_demolished = packet.game_cars[index].is_demolished,
                has_wheel_contact = packet.game_cars[index].has_wheel_contact,
                is_super_sonic = packet.game_cars[index].is_super_sonic, 
                has_jumped = packet.game_cars[index].jumped, 
                has_double_jumped = packet.game_cars[index].double_jumped, 
                boost_amount = packet.game_cars[index].boost)

    @staticmethod
    def from_pandas_frame_to_car_state(df: pd.DataFrame):
        """ changes a portion of a pandas data frame (the state at a given index) into a car state

        Args:
            df (pd.DataFrame): an indexed pandas dataframe, with columns produced by gametickpacket 

        Returns:
            [type]: [description]
        """       
        return CarState(position = Base3DVector(np.array([df.loc['location_x'], df.loc['location_y'], 0.0])),
            velocity = Base3DVector(np.array([df.loc["velocity_x"], df.loc["velocity_y"], 0.0])),
            ang_vel = Base3DVector(np.array([0,0,df.loc["angular_velocity_z"]])),
            orientation = np.array([df.loc["quaternion_w"], df.loc["quaternion_x"],df.loc["quaternion_y"],df.loc["quaternion_z"]]),
            time = df.loc["time"],
            hit_box = None,
            is_demolished = None,
            has_wheel_contact = None,
            is_super_sonic = None, 
            has_jumped = None, 
            has_double_jumped = None, 
            boost_amount = None)    


if __name__ == "__main__":
    s = StateTransformer()
    s.from_game_tick_packet_to_ball_state(GameTickPacket())