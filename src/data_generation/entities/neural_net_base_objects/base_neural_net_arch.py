# a neural net arch describes the arch (lstm vs dense)
# as well as how the model shape of the arch changes (number of hidden layers, number of neurons per hidden layer)
# activation function
# loss function

# NB: as of right now we will have a one to one mapping with a strategy and as such an arch will similarly need a factory which produces a concrete arch based on the strategy
    # - however, we possibly forsee the need to switch out different archs for the same strategy, which will call for an abstract factory, to produce a specific arch based
    # - off of a strategy and an unforseen second parameter 

    
from DeepToot.src.data_generation.entities.exceptions.implementation_exception import ImplementationException

class BaseNeuralNetArch():
    def input_shape(self):
        """
        Returns:
            Numpy.array?, tuple? -- the shape of the network (number of layers, neurons per layer)
            TODO: figure out the exact type this returns
        """     
        raise ImplementationException("You must implement your own input_shape method")    

    def output_shape(self):
        """
        Returns:
            numpy.array?, tuple? -- shape of the labels used to train the         
            TODO: figure out the exact type this returnsnetwork
            
        """ 
        raise ImplementationException("You must implement your own output_shape method")

    def shape(self):
        """
            Returns:
                Numpy.array? -- the shape of the network
                TODO: figure out the exact type this returns
        """        
        raise ImplementationException("You must implement your own shape method")      

    def type(self):
        """
        Returns:
            LSTM vs Dense vs Convolutional
            TODO: figure out the exact type this returns
        """                 
        raise ImplementationException("You must implement your own type method")

    def activation_function(self):
        """
        Returns:
            Lambda? Keras activation function 
            TODO: figure out the exact type this returns
        """    
        raise ImplementationException("You must implement your own activation_function method")
    
    def loss_function(self):
        """
        Returns:
            Lambda? Keras activation function 
            TODO: figure out the exact type this returns
        """
        raise ImplementationException("You must implement your own loss_function method")    

    def optimizer(self):
        """
        Returns:
            keras.optimizer? -- kera's predefined optimizer object
            TODO: figure out the exact type this returns
        """        
        raise ImplementationException("you must implement you rown optimizer method")