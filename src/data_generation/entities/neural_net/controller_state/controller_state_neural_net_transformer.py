from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_transformer import BaseNeuralNetTransformer
import numpy as np

class ControllerStateNeuralNetTransformer(BaseNeuralNetTransformer):
    def from_game_trajectrory_to_numpy_array(self, game_trajectory: GameTrajectory, config: ControllerStateNeuralNetModel):
        """Take the game trajectory, remove unused state elements, flatten to 1D vector and return

        Args:
            game_trajectory (GameTrajectory): trajectory to be flattened for neural network
            config (ControllerStateNeuralNetModel):

        Returns:
            numpy array: flattened trajectory that holds only x pos and vel
        """        
        simplified_trajectory = np.array([])
        for state in game_trajectory.states:
            buffer = np.array([state.position.x, state.velocity.x])
            simplified_trajectory = np.append(simplified_trajectory, buffer)
        
        return simplified_trajectory