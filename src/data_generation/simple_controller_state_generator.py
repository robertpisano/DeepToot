from rlbot.agents.base_agent import SimpleControllerState
import numpy as np
from numpy import array
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from keras.models import Sequential
from keras.activations import tanh
from keras import Input
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.losses import MeanAbsoluteError
from keras.optimizers import SGD
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_transformer import ControllerStateNeuralNetTransformer

class SimpleControllerStateGenerator():
    model = Sequential()
    
    def __init__(self, length):#game_trajectory:GameTrajectory):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]
        """        
        self.model = ControllerStateNeuralNetModel(trajectory_length=length) #game_trajectory.length) 


    def generate_controller_state(self):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]

        Returns:
            SimpleControllerState: the predicted controller inputs for motion
        """        
        numpy_game_trajectory = ControllerStateNeuralNetTransformer.from_game_trajectory_to_numpy(game_trajectory = self.game_trajectory, arch = self.model)
        output = self.model.predict(numpy_game_trajectory)

        return SimpleControllerState(steer = 0.0,
                                    throttle = output,
                                    pitch = 0.0,
                                    yaw = 0.0,
                                    roll = 0.0,
                                    jump = False,
                                    boost = False,
                                    handbrake = False,
                                    use_item = False)
