import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
import gekko
from gekko import GEKKO
import csv
from mpl_toolkits.mplot3d import Axes3D

ba=991.666
kv = np.array([0, 1410, 2300]) #velocity input
ka = np.array([1600+ba, 0+ba, 0+ba]) #acceleration ouput
kv_lin = np.linspace(0, 2300, 100)
pwl = np.interp(kv_lin, kv, ka)

coeffs = np.polyfit(kv_lin, pwl, 5) #Get degree coeff
func = np.polynomial.polynomial.Polynomial(np.flip(coeffs))



plt.figure(1)
plt.plot(kv, ka, 'g*')
plt.plot(kv_lin, func(kv_lin), 'r.')

plt.show(block=True)
