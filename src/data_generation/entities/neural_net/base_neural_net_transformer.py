# a neural net transformer takes in a game trajectory and produces a specific model with a specific arch and a specific shape to the input of that arch. 
# Each combination of these is delimited to a given strategy
#  
# NB:  at this point it's unclear if a parsing strategy (finding data in the ball chasing data) is different from a neural net transformation strategy - (which does as specified above)
            # - it is currently assumed there will be a one to one mapping and both will use the same strategy classes

# this is a base class from which different strategies will extend. This is because we may
# want to cross some of the data to get a more accurate representation of a neural net model input.

from DeepToot.src.data_generation.entities.exceptions.implementation_exception import ImplementationException
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_model import BaseNeuralNetModel

class BaseNeuralNetTransformer():
    def __init__(self, game_trajectory:GameTrajectory, neural_net_model: BaseNeuralNetModel):
        self.game_trajectory = game_trajectory
        self.neural_net_model = neural_net_model

    def from_game_trajectrory_to_numpy_array(self, game_trajectory: GameTrajectory):
        """
        Returns:
            Matrix of the shape self.neural_net_model.input_shape, output_shape

        Raises:
            ImplementationException: [description]
        """        
        raise ImplementationException("You must implement your own from_game_trajectrory_to_arch method in your extending class")