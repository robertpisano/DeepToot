import numpy as np
from scipy.integrate import solve_ivp

class BouncingBallCalculator:
    def __init__(self, state0, tf, dt):
        self.g = -680
        self.e=0.8
        self.hit_ground.terminal=True
        self.hit_ground.direction=-1
        self.tn = 0
        self.tf = tf
        self.state0 = state0
        self.dt = dt

    def calculate(self):
        #initialize iterable variables
        t = np.empty((0))
        state = np.empty((6,0))
        sn=self.state0
        tn = 0
        tf = self.tf

        while(True):
            sol = solve_ivp(self.f, [tn, tf], sn, events = BouncingBallCalculator.hit_ground, max_step = self.dt)
            # append
            t = np.append(t, sol.t[:-2])
            state = np.append(state, sol.y[:,:-2], axis=1)
            # update next iteration values
            tn = sol.t[-1]
            sn = sol.y[:,-1]
            sn[5] *= -1*self.e
            
            print(sol.t[-1])
            # check if at tf
            if(sol.t[-1] >= tf):
                break
        
        # Put variables in trajectory
        self.trajectory = BallTrajectory(t, state)



    def f(self, t, state):
        return [state[3],state[4], state[5], 0, 0, self.g]
    
    @staticmethod
    def hit_ground(t, state):
        return state[2]-90

class BallTrajectory:
    def __init__(self, t, state):
        self.t = t
        self.pos = state[0:3]
        self.vel = state[3:6]

class BallInitialState:
    def __init__(self, pos, vel):
        pass

def plot(calc):
    bc = calc.trajectory
    plt.figure()
    plt.plot(bc.t, bc.pos[0], 'r.', markersize=1)
    plt.plot(bc.t, bc.pos[1], 'g.', markersize=1)
    plt.plot(bc.t, bc.pos[2], 'b.', markersize=1)
    plt.plot(bc.t, bc.vel[0], 'lightcoral', markersize=1)
    plt.plot(bc.t, bc.vel[1], 'lightgreen', markersize=1)
    plt.plot(bc.t, bc.vel[2], 'lightblue', markersize=1)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    Axes3D.plot(ax, bc.pos[0], bc.pos[1], bc.pos[2])

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    state0 = np.array([-1000,-1000,100,100,100,1500])
    dt = 0.01
    tf = 14
    bc = BouncingBallCalculator(state0, tf, dt)
    bc.calculate()
    plot(bc)
    plt.show()
