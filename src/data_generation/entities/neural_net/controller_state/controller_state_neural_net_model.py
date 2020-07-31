
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_model import BaseNeuralNetModel

from keras.activations import sigmoid, relu, tanh
from keras.layers import Dense
from keras.losses import MeanAbsoluteError
from keras.optimizers import SGD

from keras.models import Sequential
from keras.activations import tanh
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.losses import MeanAbsoluteError
from keras.optimizers import SGD

class ControllerStateNeuralNetModel(BaseNeuralNetModel):
    def __init__(self, trajectory_length:int):
        self.trajectory_length = trajectory_length
        super(BaseNeuralNetModel, self).__init__()

        self.batch_normalization = BatchNormalization()
        self.layer_1 = Dense(self.input_shape(), activation=self.activation_function())
        self.layer_2 = Dense(self.output_shape())
        self.compile(optimizer = self.optimizer(), loss = self.loss_function(), metrics=['accuracy', 'mse'])

    def call(self, input):
        x = self.batch_normalization(input)
        x = self.layer_1(x)
        return self.layer_2(x)

    def get_config(self):
        return {"trajectory_length": self.trajectory_length }

    def input_shape(self):
        """
        Returns:
            int -- the size of the input vector (remember this is a 1D vector)
        """
        return (6*self.trajectory_length)

    def output_shape(self):
        """
        Returns:
            integer -- the size of the output vector (remember this is a 1D vector)
            
        """ 
        return 3

    def shape(self):
        """
            Returns:
                integer -- the size of the single hidden layer
                
        """        
        return 500

    def activation_function(self):
        """
        Returns:
            keras activation function type -- tanh (hyperbolic tangent)
        """    
        return sigmoid
        
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
        return SGD(learning_rate=0.03, 
                    momentum=0.0, 
                    nesterov=False, 
                    name="SGD")