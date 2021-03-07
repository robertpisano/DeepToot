import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

# Initialize gekko model
m = GEKKO()
# Number of collocation nodes
nodes = 5

# Number of phases
n = 5

# Time horizon (for all phases)
m.time = np.linspace(0,1,100)

# Input (constant in IMODE 4)
u = [m.Var(1,lb=-2,ub=2,fixed_initial=False) for i in range(n)]

# Example of same parameter for each phase
tau = 5

# Example of different parameters for each phase
K = [2,3,5,1,4]

# Scale time of each phase
tf = [1,1,1,1,1]

# Variables (one version of x for each phase)
x = [m.Var(0) for i in range(5)]

# Equations (different for each phase)
for i in range(n):
    m.Equation(tau*x[i].dt()/tf[i]==-x[i]+K[i]*u[i])

# Connect phases together at endpoints
for i in range(n-1):
    m.Connection(x[i+1],x[i],1,len(m.time)-1,1,nodes)
    m.Connection(x[i+1],'CALCULATED',pos1=1,node1=1)

# Objective
# Maximize final x while keeping third phase = -1
m.Obj(-x[n-1]+(x[2]+1)**2*100)

# Solver options
m.options.IMODE = 6
m.options.NODES = nodes

# Solve
m.solve()

# Calculate the start time of each phase
ts = [0]
for i in range(n-1):
    ts.append(ts[i] + tf[i])

# Plot
plt.figure()
tm = np.empty(len(m.time))
for i in range(n):
    tm = m.time * tf[i] + ts[i]
    plt.plot(tm,x[i])

plt.show(block=True)