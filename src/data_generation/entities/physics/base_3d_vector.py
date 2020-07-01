import numpy as np

class Base3DVector():
    vect = []
    
    def __init__(self, numpy_array=np.zeros(3, dtype=np.float)):
        self.vect = numpy_array

    def x(self):
        return self.vect[0]

    def y(self):
        return self.vect[1]

    def z(self):
        return self.vect[2]