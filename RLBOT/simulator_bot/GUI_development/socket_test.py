import numpy as np

class TestSocketClass:
    def __init__(self):
        self.data = np.linspace(0, 1, 100000)
    
    def print_data(self):
        print(self.data)
        print('size: ' + str(self.data.shape))
    