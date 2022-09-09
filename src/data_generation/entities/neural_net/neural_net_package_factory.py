
import numpy as np
from numpy import array
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_transformer import ControllerStateNeuralNetTransformer
from DeepToot.src.repositories.neural_net_model_repository import NeuralNetModelRepository
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_model import BaseNeuralNetModel
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_transformer import BaseNeuralNetTransformer



class NeuralNetPackage():
    def __init__(self, model: BaseNeuralNetModel, transformer: BaseNeuralNetTransformer):
        self.transformer = transformer
        self.model = model

class NeuralNetPackageFactory():
    def __init__(self, game_trajectory: GameTrajectory):
        """[summary]

        Args:
            game_trajectory (GameTrajectory): [description]
        """       
        self.game_trajectory = game_trajectory

    def create(self, model_type, params):
        """[summary]
        """
        if model_type == "controller":
            model_class = ControllerStateNeuralNetModel(trajectory_length=params["length"])
            model_transformer = ControllerStateNeuralNetTransformer(game_trajectory = self.game_trajectory)
        
        model = NeuralNetModelRepository(model=model_class).load()

        return NeuralNetPackage(model=model, transformer=model_transformer)

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