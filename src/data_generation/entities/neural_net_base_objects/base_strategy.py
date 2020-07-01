# a strategy currently determines both how we will parse ball chasing data for a specific play as well as how the neural net arch we will feed the data into\
# NB: we may need two separate strategies to do the above

class BaseStrategy():
    @classmethod
    def type(self):
        """
        Return:
            String - the string denoting the given type
        """
        raise ImplementationException("You must implement your own type method in your extending class")