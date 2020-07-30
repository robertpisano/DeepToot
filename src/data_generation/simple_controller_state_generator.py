from rlbot.agents.base_agent import SimpleControllerState
import numpy as np
from numpy import array
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from  keras.models import Sequential
from  keras.activations import tanh
from  keras import Input
from  keras.layers import Dense, BatchNormalization, Dropout, Activation
from  keras.losses import MeanAbsoluteError
from  keras.optimizers import SGD
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_transformer import ControllerStateNeuralNetTransformer
from DeepToot.src.repositories.neural_net_model_repository import NeuralNetModelRepository
import pandas as pd

class SimpleControllerStateGenerator():
    model = Sequential()
    
    def __init__(self, length: int):#game_trajectory:GameTrajectory):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]
        """        
        self.model_repository = NeuralNetModelRepository(model=ControllerStateNeuralNetModel(trajectory_length=length)) #game_trajectory.length) 
        self.model = self.model_repository.load()


    def generate_controller_state(self, game_trajectory: GameTrajectory):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]

        Returns:
            SimpleControllerState: the predicted controller inputs for motion
        """        
        numpy_game_trajectory = ControllerStateNeuralNetTransformer.from_game_trajectory_to_numpy_array(game_trajectory = game_trajectory, model = self.model)
        print(numpy_game_trajectory)
        output = self.model.predict(numpy_game_trajectory)

        print("generateed output" + str(output))
        return SimpleControllerState(steer = 0.0,
                                    throttle = output,
                                    pitch = 0.0,
                                    yaw = 0.0,
                                    roll = 0.0,
                                    jump = False,
                                    boost = False,
                                    handbrake = False,
                                    use_item = False)

    @staticmethod
    def from_data_frame(df: pd.DataFrame):
        return SimpleControllerState(steer = df.loc['controller_steer'],
                                    throttle = df.loc['controller_throttle'],
                                    pitch = 0.0,
                                    yaw = 0.0,
                                    roll = 0.0,
                                    jump = False,
                                    boost = df.loc['controller_boost'],
                                    handbrake = False,
                                    use_item = False)