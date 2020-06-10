from rlbot.utils.structures.game_data_struct import GameTickPacket
from DeepToot.src.entities.state.ball_state import BallState
from DeepToot.src.entities.state.car_state import CarState


# StateTransformer will convert "raw data" types (ex: GameTickPacket, pandas.DataFrame) into 
# respective state class for the trajectory builder
class StateTransformer():
    def from_game_tick_packet_to_ball_state(self, packet: GameTickPacket):
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

    def from_game_tick_packet_to_car_state(self, packet: GameTickPacket, index: int):
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
                hit_box = packet.game_info.game_cars[index].hitbox, 
                is_demolished = packet.game_info.game_cars[index].is_demolished, 
                has_wheel_contact = packet.game_info.game_cars[index].has_wheel_contact, 
                is_super_sonic = packet.game_info.game_cars[index].is_super_sonic, 
                has_jumped = packet.game_info.game_cars[index].has_jumped, 
                has_double_jumped = packet.game_info.game_cars[index].double_jumped, 
                boost_amount = packet.game_info.game_cars[index].boost)


if __name__ == "__main__":
    s = StateTransformer()
    s.from_game_tick_packet_to_ball_state(GameTickPacket())