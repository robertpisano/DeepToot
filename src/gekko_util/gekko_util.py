# List of useful classes that can inject equations into the optimizer
# Trying to make a somewhat "plug and play" methodology for building
# an optimization objective and describing kinematics and dynamics in
# a plug and play fashion as well

from gekko import GEKKO
import numpy as np
from rlbot.utils.game_state_util import CarState, BallState, Vector3, Physics, Rotator
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko
from pyquaternion import Quaternion
from DeepToot.src.dynamics_util.RollingBallCalculator import RollingBallCalculator

class Conditions():
    def __init__(self, s=Vector3(0,0,0), v_mag=0.0, r=Rotator(0,0,0), w=Vector3(0,0,0), bs=Vector3(0,0,0), bv=Vector3(0,0,0), br=Rotator(0,0,0), bw=Vector3(0,0,0)):
        # Set car state stuff
        self.s = s
        self.v_mag =v_mag
        self.r = r
        self.w = w
        yaw = r.yaw
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
        bs = Vector3(float(i['bxi']), float(i['byi']), float(i['bzi']))
        bv = Vector3(float(i['bvxi']), float(i['bvyi']), float(i['bvzi']))
        condit = Conditions(s=s, v_mag=v, r=r, bs=bs, bv=bv)
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
class AerialOptimizer(GEKKO):
    def __init__(self, ic: Conditions, fc: Conditions, num_nodes = 51):
        super().__init__(remote=False)
        self.initialConditions = ic
        self.finalConditions = fc
        self.car = GekkoFlyingCarState()
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

class RollingOptimizer(GEKKO):
    def __init__(self, ic: Conditions, fc: Conditions, num_nodes = 11):
        super().__init__(remote=False)
        self.initialConditions = ic
        self.finalConditions = fc
        self.car = GekkoDrivingCarState()
        self.ball = GekkoBallRollingState()
        
        # Setting Options first fixed no solution bug!!!
        self.set_options()

        # Time variables for simulation
        # tf: final time variable to optimize for minimizing final time
        # time: normalized time vector with num_nodes discretations
        self.tf = self.FV(value = 1, lb=1, ub=10, name='tf')
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
        self = self.ball.inject_variables(self)
        self = self.ball.inject_dynamics(self)

    def set_objectives(self, fc):
        self = GekkoOptimalDefend.inject_objectives(self)

class GekkoVector():
    def init(self, o:Optimizer, x=0, y=0, z=0, prefix='', suffix=''):
        self.x = o.Var(value = x, name=prefix+'x'+suffix)
        self.y = o.Var(value = y, name=prefix+'y'+suffix)
        self.z = o.Var(value = z, name=prefix+'z'+suffix)

        return o

class GekkoQuaternion():
    def init(self, o: Optimizer, w=1, i=0, j=0, k=0):
        self.q = o.Array(o.Var, 4) #orientation quaternion
        self.q[0].value = w
        self.q[1].value = i
        self.q[2].value = j
        self.q[3].value = k
        return o

    def inject_normalization_equations(self, o: Optimizer):
        pass

class GekkoRotation():
    def init(self, o: Optimizer, r=0, p=0, y=0, prefix='', suffix=''):
        self.roll = o.Var(value = r, name= prefix + 'roll' + suffix)
        self.pitch = o.Var(value = p, name= prefix + 'pitch' + suffix)
        self.yaw = o.Var(value = y, name= prefix + 'yaw' + suffix)
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
    def __init__(self, prefix='', suffix=''):
        self.prefix = prefix
        self.suffix = suffix
    
    def inject_variables(self, o: Optimizer):
        if(self.suffix == '1'):
            s = o.initialConditions1.s
            v = o.initialConditions1.v
            r = o.initialConditions1.r
            w = o.initialConditions1.w
            v_mag = o.initialConditions1.v_mag
        if(self.suffix == '2'):
            s = o.initialConditions2.s
            v = o.initialConditions2.v
            r = o.initialConditions2.r
            w = o.initialConditions2.w
            v_mag = o.initialConditions2.v_mag

        # State variables
        self.pos = GekkoVector()
        o = self.pos.init(o, s.x, s.y, s.z, prefix='s', suffix = self.suffix)
        self.vel = GekkoVector()
        o = self.vel.init(o, v.x, v.y, v.z, prefix='v', suffix = self.suffix)
        self.v_mag = o.Var(value = v_mag, ub = 2300, lb=0, name='v_mag' + self.suffix)
        self.orientation = GekkoRotation()
        o = self.orientation.init(o, r=0, p=0, y=r.yaw, suffix = self.suffix)
        self.ang_vel = GekkoVector()
        o = self.ang_vel.init(o, w.x, w.y, w.z, prefix='w', suffix = self.suffix)

        # Control Input Variables
        self.a = o.MV(fixed_initial=False, lb=-1, ub=1, name='a' + self.suffix) # Acceleration amount
        self.a.STATUS = 1
        self.u_turn = o.MV(fixed_initial=False, lb = -1, ub = 1, name='uturn' + self.suffix) # Steering input
        self.u_turn.STATUS = 1

        # Other dynamic variables
        #curvature vs vel as a cubic spline
        cur = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_cur = np.array([0,500,1000,1500,1750,2300])
        v_cur_fine = np.linspace(0,2300,100)
        cur_fine = np.interp(v_cur_fine, v_cur, cur)
        self.curvature = o.Var(name='curvature' + self.suffix)
        o.cspline(self.v_mag, self.curvature, v_cur_fine, cur_fine)

        ba=991.666 #acceleration due to boost
        kv = np.array([0, 1400, 1410, 2300]) #velocity input
        ka = np.array([1600+ba, 160+ba, 0+ba, 0+ba]) #acceleration ouput
        kv_fine = np.linspace(0, 2300, 10) # Finer detail so cspline can fit
        ka_fine = np.interp(kv_fine, kv, ka) # finder detail acceleration
        self.kv_fine = kv_fine
        self.ka_fine = ka_fine

        self.throttle_acceleration = o.Var(fixed_initial=False, name='throttle_accel' + self.suffix)
        o.cspline(self.v_mag, self.throttle_acceleration, kv_fine, ka_fine)
        
        return o
       


    def inject_dynamics(self, o: Optimizer):
        if(self.suffix == '1'):
            o.Equation(self.v_mag.dt()/o.tf1 == self.a * self.throttle_acceleration)
            o.Equation(self.vel.x == self.v_mag * o.cos(self.orientation.yaw))
            o.Equation(self.vel.y == self.v_mag * o.sin(self.orientation.yaw))
            o.Equation(self.pos.x.dt()/o.tf1 == self.vel.x)
            o.Equation(self.pos.y.dt()/o.tf1 == self.vel.y)
            o.Equation(self.pos.z.dt()/o.tf1 == 0.0)
            o.Equation(self.ang_vel.z == (self.u_turn * self.curvature * self.v_mag))
            o.Equation(self.orientation.yaw.dt()/o.tf1 == self.ang_vel.z)       
        if(self.suffix == '2'):
            o.Equation(self.v_mag.dt()/o.tf2 == self.a * self.throttle_acceleration)
            o.Equation(self.vel.x == self.v_mag * o.cos(self.orientation.yaw))
            o.Equation(self.vel.y == self.v_mag * o.sin(self.orientation.yaw))
            o.Equation(self.pos.x.dt()/o.tf2 == self.vel.x)
            o.Equation(self.pos.y.dt()/o.tf2 == self.vel.y)
            o.Equation(self.pos.z.dt()/o.tf2 == 0.0)
            o.Equation(self.ang_vel.z == (self.u_turn * self.curvature * self.v_mag))
            o.Equation(self.orientation.yaw.dt()/o.tf2 == self.ang_vel.z)       
        return o 

class GekkoFlyingCarState(GekkoState):
    def __init__(self):
        pass
    
    def inject_variables(self, o: Optimizer):
        s = o.initialConditions.s
        v = o.initialConditions.v
        r = o.initialConditions.r
        q = o.initialConditions.q
        w = o.initialConditions.w
        v_mag = o.initialConditions.v_mag
    
        # State variables
        self.pos = GekkoVector()
        o = self.pos.init(o, s.x, s.y, s.z, prefix='s')
        
        self.vel = GekkoVector()
        o = self.vel.init(o, v.x, v.y, v.z, prefix='v')
        
        self.v_mag = o.Var(value = v_mag, ub = 2300, lb=0, name='v_mag')
        
        self.q = o.Array(o.Var, 4) #orientation quaternion
        self.q[0].value = q.w
        self.q[1].value = q.i
        self.q[2].value = q.j
        self.q[3].value = q.k

        self.q_norm = o.Array(o.Var, 4) #orientation quaternion
        self.q_norm[0].value = q.w
        self.q_norm[1].value = q.i
        self.q_norm[2].value = q.j
        self.q_norm[3].value = q.k

        # Intermediate value since rotating vector by quateiron requires q*v*q^-1 (this is q*v)
        ux = [0,1,0,0]
        self.hi = [None, None, None, None]
        self.hi[0] = o.Intermediate(equation = -1* (self.q_norm[1]*ux[1]))
        self.hi[1] = o.Intermediate(equation = (self.q_norm[0]*ux[1]))
        self.hi[2] = o.Intermediate(equation = (self.q_norm[3]*ux[1]))
        self.hi[3] = o.Intermediate(equation =  -1* (self.q_norm[2]*ux[1]))
        
        h = q.rotate(ux[1:])
        self.heading = o.Array(o.Var, 3)
        self.heading[0].value = h[0]
        self.heading[1].value = h[1]
        self.heading[2].value = h[2]

        self.ang_vel = GekkoVector()
        o = self.ang_vel.init(o, w.x, w.y, w.z, prefix='w')

        # Control Input Variables
        self.a = o.MV(fixed_initial=False, lb=0, ub=1, name='a') # Acceleration amount
        self.a.STATUS = 1
        o.free(self.a)
        # Torque input <Tx, Ty, Tz>, will change the q_omega[1:3] values since q_omega[0] is always zero in pure quaternion form
        self.alpha = o.Array(o.MV, 3)
        for i in self.alpha:
            # i.value = 0
            i.STATUS = 1
            i.upper = 1
            i.lower = -1
            i.DCOST = 0
            o.free(i)
        # Angular Velocity input (testing for now, maybe is quicker than a torque input)
        self.q_omega = o.Array(o.Var, 4) # w, x, y, z for quaternion
        for qi in self.q_omega:
            # qi.value = 0
            # qi.STATUS = 1
            qi.upper = 1
            qi.lower = -1
            # qi.DCOST = 1e-5
        # self.q_omega[0].status = 0# Shut off scalar part of omega quaternion
        self.omega_mag = o.Var() # Magnitude of angular velocity

        # gravity
        self.g = o.Param(value = -650)
        self.D_b = o.Param(value = -0.0305) # Air drag parameter on ball

        # Drag and torque coefficient
        self.T_r = o.Param(value = -36.07956616966136) # torque coefficient for roll
        self.T_p = o.Param(value = -12.14599781908070) # torque coefficient for pitch
        self.T_y = o.Param(value = 8.91962804287785) # torque coefficient for yaw
        self.D_r = o.Param(value = -4.47166302201591) # drag coefficient for roll
        self.D_p = o.Param(value = -2.798194258050845) # drag coefficient for pitch
        self.D_y = o.Param(value = -1.886491900437232) # drag coefficient for yaw
        self.amax = o.Param(value = 991.666+60)
        self.vmax = o.Param(value = 2300)
        return o
       


    def inject_dynamics(self, o: Optimizer):
# Differental equations

        # Heading equations assuming original heading vector is <1,0,0>

        # Quaternion Derivative, q/dt = w*q
        o.Equation(self.q[0].dt() == 0.5 * o.tf * ((self.q_omega[0]*self.q[0]) - (self.q_omega[1]*self.q[1]) - (self.q_omega[2]*self.q[2]) -  (self.q_omega[3]*self.q[3])))
        o.Equation(self.q[1].dt() == 0.5 * o.tf * ((self.q_omega[0]*self.q[1]) + (self.q_omega[1]*self.q[0]) + (self.q_omega[2]*self.q[3]) - (self.q_omega[3]*self.q[2])))
        o.Equation(self.q[2].dt() == 0.5 * o.tf * ((self.q_omega[0]*self.q[2]) - (self.q_omega[1]*self.q[3]) + (self.q_omega[2]*self.q[0]) + (self.q_omega[3]*self.q[1])))
        o.Equation(self.q[3].dt() == 0.5 * o.tf * ((self.q_omega[0]*self.q[3]) + (self.q_omega[1]*self.q[2]) - (self.q_omega[2]*self.q[1]) + (self.q_omega[3]*self.q[0])))

        # Normalize quaternion after it has been affected by possible noise from quaternion derivative floating point math
        o.Equation(self.q_norm[0] == self.q[0]/o.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
        o.Equation(self.q_norm[1] == self.q[1]/o.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
        o.Equation(self.q_norm[2] == self.q[2]/o.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
        o.Equation(self.q_norm[3] == self.q[3]/o.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))

        # Omega/dt, angular acceleration, alpha are the controller inputs, and their related Torque magnitude (roll, pitch, yaw | T_r, T_p, T_y)
        # There is also a damper component to the rotataion dynamics, for pitch and yaw, the dampening effect is altered by the input value of that
        # rotational input, and requres an absoulte value but to make calculation converge better I tried sqrt(var^2) instead of the absoulte value
        # Since absoulte value would require me to move to integer type soltuion
        o.Equation(self.q_omega[1].dt() == o.tf * ((self.alpha[0] * self.T_r) + (self.q_omega[1]*self.D_r)))
        o.Equation(self.q_omega[2].dt() == o.tf * ((self.alpha[1] * self.T_p) + (self.q_omega[2]*self.D_p * (1-o.sqrt(self.alpha[1]*self.alpha[1])))))
        o.Equation(self.q_omega[3].dt() == o.tf * ((self.alpha[2] * self.T_y) + (self.q_omega[3]*self.D_y * (1-o.sqrt(self.alpha[2]*self.alpha[2])))))
        # o.Equation(self.q_omega[1].dt() == self.tf * (0.5 * self.alpha[0] * self.T_r)) # I'm multiplying the acceleration output by 0.5 to keep paths reachable by feedback controller only
        # o.Equation(self.q_omega[2].dt() == self.tf * (0.5 * self.alpha[1] * self.T_p))
        # o.Equation(self.q_omega[3].dt() == self.tf * (0.5 * self.alpha[2] * self.T_y))


        # Get unit vector pointing in heading direction (hi is q*v, this is (q*v)*q^-1) Thats why it has negatives on q_norm[1,2,3]
        # hi is the intermediate q*v, which is the lefthand side of the vector equation (q*v)*q^-1 to rotate a vector by the quaternion
        # This equation is the right handsize of the equation (q*v)*q^-1
        # hi is always rotating the quaternion amount from the unit x vector [0, 1, 0, 0] [w, i, j, k]
        # Unit x vector
        ux = [0, 1, 0, 0]
        o.Equation(self.heading[0] == (self.hi[0]*-1*self.q_norm[1]) + (self.hi[1]*self.q_norm[0]) + (self.hi[2]*-1*self.q_norm[3]) - (self.hi[3]*-1*self.q_norm[2]))
        o.Equation(self.heading[1] == (self.hi[0]*-1*self.q_norm[2]) - (self.hi[1]*-1*self.q_norm[3]) + (self.hi[2]*self.q_norm[0]) + (self.hi[3]*-1*self.q_norm[1]))
        o.Equation(self.heading[2] == (self.hi[0]*-1*self.q_norm[3]) + (self.hi[1]*-1*self.q_norm[2]) - (self.hi[2]*-1*self.q_norm[1]) + (self.hi[3]*self.q_norm[0]))

        o.Equation(self.v_mag == o.sqrt((self.vel.x)**2 + (self.vel.y)**2 + (self.vel.z)**2))
        o.Equation(self.v_mag <= 2300)

        o.Equation(self.vel.x.dt() == o.tf * self.amax * self.a * self.heading[0])
        o.Equation(self.vel.y.dt()== o.tf * self.amax * self.a * self.heading[1])
        o.Equation(self.vel.z.dt() == o.tf * ((self.amax * self.a * self.heading[2]) + self.g))
        # Car velocity with drag componnent
        # o.Equation(self.vx.dt() == self.tf * ((self.amax * self.a * self.heading[0]) + (self.vx * self.D_b)))
        # o.Equation(self.vy.dt() == self.tf * ((self.amax * self.a * self.heading[1]) + (self.vy * self.D_b)))
        # o.Equation(self.vz.dt() == self.tf * ((self.amax * self.a * self.heading[2]) + self.g + (self.vz * self.D_b)))

        o.Equation(self.pos.x.dt() == o.tf * self.vel.x)
        o.Equation(self.pos.y.dt() == o.tf * self.vel.y)
        o.Equation(self.pos.z.dt() == o.tf * self.vel.z)

        # Limit maximum total omega
        o.Equation(self.omega_mag == o.sqrt((self.q_omega[1]**2) + (self.q_omega[2]**2) + self.q_omega[3]**2))
        o.Equation(self.omega_mag <= 5.5)

        return o 


class GekkoBallState(GekkoState):
    pass
class GekkoBallSplineState(RollingBallCalculator, GekkoState):
    def __init__(self):
        pass

    def inject_variables(self, o: Optimizer):
        s = o.initialConditions1.bs
        v = o.initialConditions1.bv
        state0 = np.array([s.x,s.y,s.z,v.x,v.y,v.z])
        tf = o.tf1.upper
        dt = 0.1
        super().__init__(state0, tf, dt)
        self.calculate()
        ball = self.trajectory

        #state variables
        self.bs1 = o.Array(o.Var, (3)) # ball position1
        for ind, b in enumerate(self.bs1):
            b.value = ball.pos[ind,0]
        self.bv1 = o.Array(o.Var, (3)) # ball velocity1
        for ind, b in enumerate(self.bv1):
            b.value = ball.vel[ind,0]
        self.bs2 = o.Array(o.Var, (3)) # ball position2
        for ind, b in enumerate(self.bs2):
            b.value = ball.pos[ind,0]
        self.bv2 = o.Array(o.Var, (3)) # ball velocity2
        for ind, b in enumerate(self.bv2):
            b.value = ball.vel[ind,0]

        # c-spline of ball trajectory
        for index, i in enumerate(self.bs1):
            o.cspline(o.solver_time1, i, ball.t, ball.pos[index])
        for index, i in enumerate(self.bv1):
            o.cspline(o.solver_time1, i, ball.t, ball.vel[index])
        # c-spline of ball trajectory
        for index, i in enumerate(self.bs2):
            o.cspline(o.solver_time2, i, ball.t, ball.pos[index])
        for index, i in enumerate(self.bv2):
                 o.cspline(o.solver_time2, i, ball.t, ball.vel[index])





class GekkoBallRollingState(GekkoState):
    def __init__(self):
        self.D_b = -0.0305 # Air drag parameter on ball

        pass
    
    def inject_variables(self, o: Optimizer):
        s = o.initialConditions1.bs
        v = o.initialConditions1.bv
        # w = o.initialConditions.w
    
        # State variables
        self.pos = GekkoVector()
        o = self.pos.init(o, s.x, s.y, s.z, prefix='bs')
        self.vel = GekkoVector()
        o = self.vel.init(o, v.x, v.y, v.z, prefix='bv')

        return o
       


    def inject_dynamics(self, o: Optimizer):
        o.Equation(self.vel.x.dt()/o.tf == self.D_b*self.vel.x)
        o.Equation(self.vel.y.dt()/o.tf == self.D_b*self.vel.y)
        o.Equation(self.pos.x.dt()/o.tf == self.vel.x)
        o.Equation(self.pos.y.dt()/o.tf == self.vel.y)
        o.Equation(self.pos.z.dt()/o.tf == 0.0)    

        return o 

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

class GekkoOptimalDefend(GekkoObjective):
    @staticmethod
    def inject_objectives(o: Optimizer):
        o.Minimize(o.tf * 1e3)
        o.Minimize(o.final * 1e8 * ((o.car.pos.x - o.ball.pos.x)**2))        
        o.Minimize(o.final * 1e8 * ((o.car.pos.y - o.ball.pos.y)**2))
        # o.Maximize(o.final * (o.car.v_mag**2))
        o.Minimize(o.final * (((np.pi/2) - o.car.orientation.yaw)**2) * 1e6)

class GekkoMinimumTimeToBallInAir(GekkoObjective):
    pass

class GekkoMinimumTimeComparison(GekkoObjective):
    @staticmethod
    def inject_objectives(o: Optimizer):
        o.Minimize(o.final * 1e4 * ((o.c1.pos.x - o.ball.bs1[0])**2))
        o.Minimize(o.final * 1e4 * ((o.c2.pos.x - o.ball.bs2[0])**2))
        o.Minimize(o.final * 1e4 * ((o.c1.pos.y - o.ball.bs1[1])**2))
        o.Minimize(o.final * 1e4 * ((o.c2.pos.y - o.ball.bs2[1])**2))