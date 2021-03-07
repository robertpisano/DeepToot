from gekko import brain
import struct
import numpy as np

# Get data from binary file

file = open("D:/Documents/RLUtilities/assets/soccar/soccar_navigation_nodes.bin",
            "rb")
x = []
y = []
z = []
byte = file.read(4)
while(byte):
    x.append(list(struct.unpack('f', byte)))
    byte = file.read(4)
    y.append(list(struct.unpack('f', byte)))
    byte = file.read(4)
    z.append(list(struct.unpack('f', byte)))
    byte = file.read(4)

file = open("D:/Documents/RLUtilities/assets/soccar/soccar_navigation_normals.bin",
            "rb")
xn = []
yn = []
zn = []
byte = file.read(4)
while(byte):
    xn.append(list(struct.unpack('f', byte)))
    byte = file.read(4)
    yn.append(list(struct.unpack('f', byte)))
    byte = file.read(4)
    zn.append(list(struct.unpack('f', byte)))
    byte = file.read(4)

len(x), len(y), len(z), len(xn), len(yn), len(zn)

# setup input and output matricies
# dim(axis) 0 is input layer size (3), and dim(axis) 1 is #datasets
# Transpose since vector comes in with axis 0 the data sets
x = np.asarray(x).T
y = np.asarray(y).T
z = np.asarray(z).T
xn = np.asarray(xn).T
yn = np.asarray(yn).T
zn = np.asarray(zn).T
x.shape, y.shape, z.shape, xn.shape, yn.shape, zn.shape

input = np.concatenate((x,y,z), axis = 0)
output = np.concatenate((xn, yn, zn), axis = 0)
input.shape

b = brain.Brain(m=[], remote = False, bfgs=True, explicit = False)

b.input_layer(3)

b.layer(ltype = 'dense', linear=100)
b.layer(ltype = 'dense', tanh=100)
b.layer(ltype = 'dense', linear=100)

b.output_layer(3, ltype='dense', activation='linear')

b.learn(input[:,:10], output[:,:10], obj=2, gap=0.01, disp=True)





