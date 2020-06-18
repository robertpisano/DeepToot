from rlbot.agents.base_agent import SimpleControllerState
import numpy as np
from numpy import array
from DeepToot.src.entities.controller_state_net_architecture import ControllerStateNetworkArchitecture
from DeepToot.src.entities.physics.game_trajectory import GameTrajectory
from keras.models import Sequential
from keras.activations import tanh
from keras.layers import Dense
from keras.losses import MeanAbsoluteError
from keras.optimizers import SGD
from DeepToot.src.entities.controller_state_neural_net_transformers import ControllerStateNeuralNetTransformer

class SimpleControllerStateGenerator():
    model = Sequential()
    
    def __init__(self, game_trajectory:GameTrajectory):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]
        """        
        self.configuration = ControllerStateNetworkArchitecture(trajectory_length=game_trajectory.length) 
        self.model.add(Dense(self.configuration.output_shape(), 
                                input_shape = self.configuration.input_shape(),
                                use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(self.configuration.activation_function())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.configuration.output_shape()))
        self.model.compile(optimizer = self.configuration.optimizer(), loss = self.configuration.loss_function, metrics=['accuracy', 'mse'])
        self.game_trajectory = game_trajectory



    def generate_controller_state(self):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]

        Returns:
            SimpleControllerState: the predicted controller inputs for motion
        """        
        numpy_game_trajectory = ControllerStateNeuralNetTransformer.from_game_trajectory_to_numpy(game_trajectory = self.game_trajectory, architecture = self.configuration)
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
