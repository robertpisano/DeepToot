from DeepToot.src.entities.neural_net_architectures.controller_state_neural_net_architecture import ControllerStateNeuralNetArch
from DeepToot.src.entities.physics.game_trajectory import GameTrajectory
import numpy as np

class ControllerStateNeuralNetTransformer(BaseNeuralNetTransformer):
    def from_game_trajectrory_to_numpy_array(self, game_trajectory: GameTrajectory, config: ControllerStateNeuralNetArch):
        """Take the game trajectory, remove unused state elements, flatten to 1D vector and return

        Args:
            game_trajectory (GameTrajectory): trajectory to be flattened for neural network
            config (ControllerStateNeuralNetArch):

        Returns:
            numpy array: flattened trajectory that holds only x pos and vel
        """        
        simplified_trajectory = np.array([])
        for state in game_trajectory.states:
            buffer = np.array([state.position.x, state.velocity.x])
            simplified_trajectory = np.append(simplified_trajectory, buffer)
        
        return simplified_trajectory