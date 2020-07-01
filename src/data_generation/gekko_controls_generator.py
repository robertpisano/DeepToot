import numpy as np
from gekko import GEKKO
from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory
from DeepToot.src.data_generation.entities.state.car_state import CarState, CarStateBuilder

from DeepToot.src.data_generation.entities.physics.base_3d_vector import Base3DVector

import sys
import linecache
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

class GekkoControlsGenerator:
    def __init__(self, length):
        self.generator = GEKKO(remote=False)
        self.num_segments = length # How many elements the time vector has
        self.generator.time = np.linspace(0, 1, self.num_segments)

        #Dynamic Optimization options
        self.generator.options.NODES = 2
        self.generator.options.SOLVER = 3
        self.generator.options.IMODE = 6
        self.generator.options.MAX_ITER = 500
        self.generator.options.MV_TYPE = 1
        self.generator.options.DIAGLEVEL = 0

        # Final time scaling variable for optimization function
        self.tf = self.generator.FV(value = 1)

        # Allow gekko to change the value of tf
        self.tf.STATUS = 0

        # Acceleration Variable
        self.a = self.generator.MV(value = 0, lb = -1, ub = 1)
        self.a.STATUS = 1
        self.a.DCOST = 0.000001


        # Max acceleration
        self.amax = self.generator.Param(value = 900)

    def generate_controls(self, traj: Trajectory):
        try:
            # Convert trajectory into array
            self.position_trajectory = []
            self.velocity_trajectory = []
            for pt in traj.states:
                self.position_trajectory = pt.position.x()
                self.velocity_trajectory = pt.velocity.x()

            # Define state variables and initial values
            self.x = self.generator.Var(value = traj.states[0].position.x())
            self.vx = self.generator.Var(value = traj.states[0].velocity.x())

            # Define differential equations
            self.generator.Equation(self.x.dt() == self.tf * (self.vx))
            self.generator.Equation(self.vx.dt() == self.tf * self.a * self.amax)

            # Define piece wise linear throttle curve
            # self.v_for_throttle = np.array([0, 1400, 1410])
            # self.a_for_throttle = np.array([1600, 160, 0])
            # self.throttle_acceleration = self.d.Var(value = 1, lb = 0, ub = 1)
            # self.d.pwl(self.v_mag, self.throttle_acceleration, self.v_for_throttle, self.a_for_throttle)


            # Objective functions
            # self.generator.Obj(1e5 * (self.position_trajectory - self.x)**2)
            self.generator.Obj(1e5 * (self.velocity_trajectory - self.vx)**2)

            self.generator.solve(disp=True)
        
        except Exception as e:
            PrintException()



def test_gekko_controls_generator():
    sarray = []
    velocities = np.linspace(0, 100, 100)
    for i in velocities:
        
        pos = np.array([0, 0, 0])
        vel = np.array([i, 0, 0])
        # ang_vel = np.array([0, 0, 0])
        # rotation = np.array([0, 0, 0])
        # state = CarState(position = pos, velocity = vel, ang_vel = ang_vel, orientation = rotation)
        state = CarStateBuilder().set_position(pos).set_velocity(vel).build()
        sarray.append(state)

    trajectory = Trajectory(state_array=sarray)
    opt = GekkoControlsGenerator(10)
    opt.generate_controls(trajectory)

    print(opt.x.value)
    print(opt.vx.value)
    print(opt.a.value)



if __name__ == "__main__":
    """Script to test training of simple controls generator network
    """    

    test_gekko_controls_generator()