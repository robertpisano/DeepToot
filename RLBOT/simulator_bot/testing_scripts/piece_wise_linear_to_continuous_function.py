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

ba=991.666 #acceleration due to boost
kv = np.array([0, 1400, 1410, 2300]) #velocity inpu,t
ka = np.array([1600+ba, 160+ba, 0+ba, 0+ba]) #acceleration ouput
kv_fine = np.linspace(0, 2300, 100)
ka_fine = np.interp(kv_fine, kv, ka)

coeffs = np.polyfit(kv_fine, ka_fine, 5) #Get degree coeff
func = np.polynomial.polynomial.Polynomial(np.flip(coeffs))

cubic = scipy.interpolate.CubicSpline(kv_fine, ka_fine)


plt.figure(1)
# plt.plot(kv, ka, 'g*')
plt.plot(kv_fine, cubic(kv_fine), 'r-')
plt.scatter(kv_fine, ka_fine)
plt.scatter(kv, ka)

plt.show(block=True)
