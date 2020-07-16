from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_transformer import BaseNeuralNetTransformer
import numpy as np

class ControllerStateNeuralNetTransformer(BaseNeuralNetTransformer):
    @staticmethod
    def from_game_trajectory_to_numpy_array(game_trajectory: GameTrajectory, model: ControllerStateNeuralNetModel):
        """Take the game trajectory, remove unused state elements, flatten to 1D vector and return

        Args:
            game_trajectory (GameTrajectory): trajectory to be flattened for neural network
            model (ControllerStateNeuralNetModel):

        Returns:
            numpy array: flattened trajectory that holds only x pos and vel
        """        
        simplified_trajectory = np.expand_dims(np.array([game_trajectory.BOT_TRAJECTORY.state_at_index(-1).velocity.y], dtype=np.float), axis=1)
        simplified_trajectory = np.append(simplified_trajectory, np.expand_dims(np.array([game_trajectory.BOT_TRAJECTORY.state_at_index(-1).velocity.y + 500], dtype=np.float), axis=1), axis = 1)
        return simplified_trajectory