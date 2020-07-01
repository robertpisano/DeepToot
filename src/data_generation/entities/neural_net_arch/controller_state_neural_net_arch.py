
from DeepToot.src.data_generation.entities.neural_net_base_objects.base_neural_net_arch import BaseNeuralNetArch 
from keras.activations import sigmoid, relu
from keras.layers import Dense
from keras.losses import MeanAbsoluteError
from keras.optimizers import SGD

class ControllerStateNeuralNetArch():
    def __init__(self, trajectory_length:int):
        self.trajectory_length = trajectory_length
        
    def input_shape(self):
        """
        Returns:
            int -- the size of the input vector (remember this is a 1D vector)
        """
        return (2*self.trajectory_length)

    def output_shape(self):
        """
        Returns:
            integer -- the size of the output vector (remember this is a 1D vector)
            
        """ 
        return 1

    def shape(self):
        """
            Returns:
                integer -- the size of the single hidden layer
                
        """        
        return 10

    def type(self):
        """
        Returns:
            layer type -- the type of neural network style (LSTM, CONVOLUTIONAL, DENSE)
        """                 
        return Dense

    def activation_function(self):
        """
        Returns:
            keras activation function type -- tanh (hyperbolic tangent)
        """    
        return relu
        
    def loss_function(self):
        """
        Returns:
            Keras loss function - mean absolute erro (mae)
        """
        return MeanAbsoluteError()

    def optimizer(self):
        """
        Returns:
            keras.optimizer -- SGD (stochastic gradient descent, learing rate / momentum parameters probably need tuning)
        """        
        return SGD(learning_rate=0.01, 
                    momentum=0.01, 
                    nesterov=False, 
                    name="SGD")