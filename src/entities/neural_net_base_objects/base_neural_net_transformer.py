# a neural net transformer takes in a game trajectory and produces a specific model with a specific architecture and a specific shape to the input of that architecture. 
# Each combination of these is delimited to a given strategy
#  
# NB:  at this point it's unclear if a parsing strategy (finding data in the ball chasing data) is different from a neural net transformation strategy - (which does as specified above)
            # - it is currently assumed there will be a one to one mapping and both will use the same strategy classes

# this is a base class from which different strategies will extend. This is because we may
# want to cross some of the data to get a more accurate representation of a neural net model input.

from DeepToot.src.entities.exceptions.implementation_exception import ImplementationException
from DeepToot.src.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.entities.neural_net_base_objects.base_neural_net_architecture import NeuralNetArch

class BaseNeuralNetTransformer():
    def __init__(self, game_trajectory:GameTrajectory, neural_net_arch: NeuralNetArch):
        self.game_trajectory = game_trajectory
        self.neural_net_arch = neural_net_arch

    def from_game_trajectrory_to_?(self):
        """
        Returns:
            Matrix of the shape self.neural_net_arch.input_shape, output_shape

        Raises:
            ImplementationException: [description]
        """        
        raise ImplementationException("You must implement your own from_game_trajectrory_to_architecture method in your extending class")