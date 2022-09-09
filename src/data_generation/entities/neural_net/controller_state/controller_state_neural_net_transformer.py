from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.state.state_transformer import StateTransformer
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_transformer import BaseNeuralNetTransformer
import numpy as np

class ControllerStateNeuralNetTransformer(BaseNeuralNetTransformer):
    def __init__(self, game_trajectory: GameTrajectory, model):
        super().__init__()
        self.game_trajectory = game_trajectory


    def from_game_trajectory_to_numpy_array(game_tick_packet, self_index, opponent_index):
        """Take the game trajectory, remove unused state elements, flatten to 1D vector and return

        Args:
            game_trajectory (GameTrajectory): trajectory to be flattened for neural network

        Returns:
            numpy array: flattened trajectory that holds only x pos and vel
        """
        simplified_trajectory = np.expand_dims(np.array([self.game_trajectory.BOT_TRAJECTORY.state_at_index(-1).velocity.y], dtype=np.float), axis=1)
        simplified_trajectory = np.append(simplified_trajectory, np.expand_dims(np.array([self.game_trajectory.BOT_TRAJECTORY.state_at_index(-1).velocity.y + 500], dtype=np.float), axis=1), axis = 1)
        return simplified_trajectory

    def label():
         # make a state vector buffer from game trajectory builder
        state_buff = [gtb.bot_queue[1].time, gtb.bot_queue[1].position.x(), gtb.bot_queue[1].position.y(), gtb.bot_queue[1].velocity.x(), gtb.bot_queue[1].velocity.y(), gtb.bot_queue[1].ang_vel.z(),
                        gtb.bot_queue[2].time, gtb.bot_queue[2].position.x(), gtb.bot_queue[2].position.y(), gtb.bot_queue[2].velocity.x(), gtb.bot_queue[2].velocity.y(), gtb.bot_queue[2].ang_vel.z()]
        
        # output data vector buffer
        control_buff = [dataframe.loc[index, 'controller_throttle'], dataframe.loc[index, 'controller_steer'], dataframe.loc[index, 'controller_boost']]
        
        # Fill the elements of the in_data, out_data arrays
        if index == 0: # Initialize in_data and out_data as empty arrays with proper size
            in_data = np.empty(shape = (data_frames_length, len(state_buff)))
            out_data = np.empty(shape = (data_frames_length, len(control_buff)))
        # Fill the arrays
        for j, elem in enumerate(state_buff):
            in_data[index][j] = elem
        for k, celem in enumerate(control_buff): 
            out_data[index][k] = celem
    
    def feature():
        