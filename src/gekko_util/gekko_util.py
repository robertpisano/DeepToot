# List of useful classes that can inject equations into the optimizer
# Trying to make a somewhat "plug and play" methodology for building
# an optimization objective and describing kinematics and dynamics in
# a plug and play fashion as well

from gekko import GEKKO
import numpy as np
from rlbot.utils.game_state_util import CarState, BallState, Vector3, Physics, Rotator
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko

class Conditions():
    def __init__(self, s=Vector3(0,0,0), v_mag=0.0, r=Rotator(0,0,0), w=Vector3(0,0,0), bs=Vector3(0,0,0), bv=Vector3(0,0,0), br=Rotator(0,0,0), bw=Vector3(0,0,0)):
        # Set car state stuff
        self.s = s
        self.v_mag =v_mag
        self.r = r
        self.w = w
        yaw = r.yaw
        print('yaw:', yaw)
        vx = v_mag * np.cos(yaw)
        vy = v_mag * np.sin(yaw)
        self.v = Vector3(x=vx, y=vy, z=0.0)

        self.bs = bs
        self.bv = bv
        self.br = br
        self.bw = bw

    @staticmethod
    def build_initial_from_initial_conditions_object(ic: InitialConditionsGekko):
        i = ic.params
        s = Vector3(float(i['sxi']), float(i['syi']), float(i['szi']))
        v = float(i['v_magi'])
        r = Rotator(roll=float(i['rolli']), pitch=float(i['pitchi']), yaw=float(i['yawi']))
        condit = Conditions(s=s, v_mag=v, r=r)
        return condit

    @staticmethod
    def build_final_from_initial_conditions_object(fc: InitialConditionsGekko):
        i = fc.params
        s = Vector3(float(i['sxf']), float(i['syf']), float(i['szf']))
        v = float(i['v_magf'])
        r = Rotator(float(i['rollf']), float(i['pitchf']), float(i['yawf']))
        condit = Conditions(s=s, v_mag=v, r=r)
        return condit

class Optimizer(GEKKO):
    def __init__(self, ic: Conditions, fc: Conditions, num_nodes = 51):
        super().__init__(remote=False)
        self.initialConditions = ic
        self.finalConditions = fc
        self.car = GekkoDrivingCarState()
        self.ball = GekkoBallState()
        
        # Setting Options first fixed no solution bug!!!
        self.set_options()

        # Time variables for simulation
        # tf: final time variable to optimize for minimizing final time
        # time: normalized time vector with num_nodes discretations
        self.tf = self.FV(value = 1, lb=1, ub=100, name='tf')
        self.tf.STATUS = 1 # Let optimizer adjust tf
        self.time = np.linspace(0, 1, num_nodes)

        self.p_d = np.zeros(num_nodes)
        self.p_d[-1] = 1.0
        self.final = self.Param(value = self.p_d, name='final')


        self.set_all_equations()
        self.set_objectives(fc)

    def set_options(self):
        self.options.NODES=3
        self.options.SOLVER=3
        self.options.IMODE=6
        self.options.MAX_ITER=800
        self.options.MV_TYPE = 0
        # self.options.DIAGLEVEL=1

    def set_all_equations(self):
        self = self.car.inject_variables(self)
        self = self.car.inject_dynamics(self)

    def set_objectives(self, fc):
        self = GekkoMinimumTimeToPosition.inject_position_objectives(self, self.finalConditions)



class GekkoVector():
    def init(self, o:Optimizer, x=0, y=0, z=0, prefix=''):
        self.x = o.Var(value = x, name=prefix+'x')
        self.y = o.Var(value = y, name=prefix+'y')
        self.z = o.Var(value = z, name=prefix+'z')

        return o

class GekkoQuaternion():
    def init(self, o: Optimizer, w=1, i=0, j=0, k=0):
        self.w = o.Var(value = w)
        self.i = o.Var(value = i)
        self.j = o.Var(value = j)
        self.k = o.Var(value = k)
        return o

    def inject_normalization_equations(self, o: Optimizer):
        pass

class GekkoRotation():
    def init(self, o: Optimizer, r=0, p=0, y=0):
        self.roll = o.Var(value = r, name='roll')
        self.pitch = o.Var(value = p, name='pitch')
        self.yaw = o.Var(value = y, name='yaw')
        return o

# GEKKO State Equations and Variables
class GekkoState():

    def __init__(self):
        equations = []
        variables = []

    @staticmethod
    def inject_dynamics(o: Optimizer):
        raise NotImplementedError
    
    @staticmethod
    def inject_variables(o: Optimizer):
        raise NotImplementedError

    @staticmethod
    def inject_other_characteristics(o: Optimizer):
        raise NotImplementedError



# GEKKO Dynamic Equations
class GekkoObjective():
    @staticmethod
    def inject_objectives(car: GekkoState, ball: GekkoState):
        pass





#Example Classes for equation/variable injection

class GekkoDrivingCarState(GekkoState):
    def __init__(self):
        pass
    
    def inject_variables(self, o: Optimizer):
        s = o.initialConditions.s
        v = o.initialConditions.v
        r = o.initialConditions.r
        w = o.initialConditions.w
        v_mag = o.initialConditions.v_mag
    
        # State variables
        self.pos = GekkoVector()
        o = self.pos.init(o, s.x, s.y, s.z, prefix='')
        self.vel = GekkoVector()
        o = self.vel.init(o, v.x, v.y, v.z, prefix='v')
        self.v_mag = o.Var(value = v_mag, ub = 2300, lb=0, name='v_mag')
        self.orientation = GekkoRotation()
        o = self.orientation.init(o, r=0, p=0, y=r.yaw)
        self.ang_vel = GekkoVector()
        o = self.ang_vel.init(o, w.x, w.y, w.z, prefix='w')

        # Control Input Variables
        self.a = o.MV(fixed_initial=False, lb=-1, ub=1, name='a') # Acceleration amount
        self.a.STATUS = 1
        self.u_turn = o.MV(fixed_initial=False, lb = -1, ub = 1, name='uturn') # Steering input
        self.u_turn.STATUS = 1

        # Other dynamic variables
        #curvature vs vel as a cubic spline
        cur = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_cur = np.array([0,500,1000,1500,1750,2300])
        v_cur_fine = np.linspace(0,2300,100)
        cur_fine = np.interp(v_cur_fine, v_cur, cur)
        self.curvature = o.Var(name='curvature')
        o.cspline(self.v_mag, self.curvature, v_cur_fine, cur_fine)

        ba=991.666 #acceleration due to boost
        kv = np.array([0, 1400, 1410, 2300]) #velocity input
        ka = np.array([1600+ba, 160+ba, 0+ba, 0+ba]) #acceleration ouput
        kv_fine = np.linspace(0, 2300, 10) # Finer detail so cspline can fit
        ka_fine = np.interp(kv_fine, kv, ka) # finder detail acceleration
        self.kv_fine = kv_fine
        self.ka_fine = ka_fine

        self.throttle_acceleration = o.Var(fixed_initial=False, name='throttle_accel')
        o.cspline(self.v_mag, self.throttle_acceleration, kv_fine, ka_fine)
        
        return o
       


    def inject_dynamics(self, o: Optimizer):
        o.Equation(self.v_mag.dt()/o.tf == self.a * self.throttle_acceleration)
        o.Equation(self.vel.x == self.v_mag * o.cos(self.orientation.yaw))
        o.Equation(self.vel.y == self.v_mag * o.sin(self.orientation.yaw))
        o.Equation(self.pos.x.dt()/o.tf == self.vel.x)
        o.Equation(self.pos.y.dt()/o.tf == self.vel.y)
        o.Equation(self.pos.z.dt()/o.tf == 0.0)
        o.Equation(self.ang_vel.z == (self.u_turn * self.curvature * self.v_mag))
        o.Equation(self.orientation.yaw.dt()/o.tf == self.ang_vel.z)       

        return o 

class GekkoBallState(GekkoState):
    pass

class GekkoMinimumTimeToPosition(GekkoObjective):
    @staticmethod
    def inject_position_objectives(o: Optimizer, final_state: CarState):
        x = final_state.s.x
        y = final_state.s.y
        o.Minimize(o.final * 1e4 * ((o.car.pos.x - x)**2))
        o.Minimize(o.final * 1e4 * ((o.car.pos.y - y)**2))
        o.Minimize(1e1 * o.tf)

        return o

class GekkoMinimumTimeToBallOnGround(GekkoObjective):
    @staticmethod
    def inject_objectives(o: Optimizer):
        o.Minimize(o.final * 1e4 * ((o.car.pos.x - o.ball.pos.x)**2))
        # o.Minimize(o.final * 1e4 )

class GekkoMinimumTimeToBouncingBall(GekkoObjective):
    @staticmethod
    def inject_objectives(o: Optimizer):
        pass