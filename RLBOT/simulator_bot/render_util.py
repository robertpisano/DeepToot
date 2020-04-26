from rlbot.agents.base_agent import BaseAgent
import numpy as np
import simulator_utilities as sim_util

def basic_render(bot, packet, traj):
    # start rendering
    bot.renderer.begin_rendering()

    # coord system render
    render_intertial_coordinate_system(bot)

    # bot reference frame coordinate system
    render_reference_coordinate_system(bot, packet)

    # desired quaternion vector
    render_desired_quaternion_vector(bot, packet, traj)

    # finish rendering
    bot.renderer.end_rendering()

def render_intertial_coordinate_system(bot: BaseAgent):

    
    # define some important coordinate features
    origin = [0,0,0]
    x = [500, 0, 0]
    y = [0, 500, 0]
    z = [0, 0, 500]

    # draw x axis in red
    bot.renderer.draw_line_3d(origin, x, bot.renderer.red())

    # draw y axis in green
    bot.renderer.draw_line_3d(origin, y, bot.renderer.green())

    # draw z axis in blue
    bot.renderer.draw_line_3d(origin, z, bot.renderer.blue())

def render_reference_coordinate_system(bot, packet):
    None

def render_desired_quaternion_vector(bot, packet, traj):
    b = bot.renderer
    loc = packet.game_cars[bot.index].physics.location
    pos = np.array([loc.x, loc.y, loc.z])
    origin = [0,0,0]
    ux = np.array([300,0,0])
    colors = [b.red(), b.blue(), b.orange(), b.green(), b.grey(), b.pink(), b.purple(), b.white(), b.cyan()]

    for i, s in enumerate(traj.states):
        ux_prime = s.orientation.unit.rotate(ux)

        # ux_prime[1] *= -1
        try:
            bot.renderer.draw_line_3d(pos, pos+ux_prime, colors[i])
        except:
            from AerialOptimization import AerialOptimizer as ao
            ao.PrintException()
            bot.renderer.draw_line_3d(pos, pos+ux_prime, colors[-1])
    
def render_math_check_vector(bot, packet):
    b = bot.renderer
    ux = np.array([300,0,0])
    loc = packet.game_cars[bot.index].physics.location
    pos = np.array([loc.x, loc.y, loc.z])
    e = packet.game_cars[bot.index].physics.orientation
    euler = np.array([e.roll, e.pitch, e.yaw])

def sanity_checking(bot, packet):
    from pyquaternion import Quaternion

    # Initialize state() from packet
    s = sim_util.State()
    s.init_from_packet(packet, bot.index)

    # Rotation matrix from euler angles
    # rot_mat = sim_util.convert_from_euler_angles(s)
    rot_mat, _ = sim_util.euler_to_rotation_to_quaternion(s)

    # Quaternion from rotation matrix
    quat = Quaternion(matrix=rot_mat).unit

    # Create coord render class
    origin = np.array([0, 0, 500])
    coord_render = CoordinateRender(origin, rot_mat, quat)

    # Render both new coordinate systems
    bias = np.array([500, 0, 0])
    coord_render.render_with_bias(bot, bias)

class CoordinateRender():

    def __init__(self, origin, rot_mat, quat):
        self.origin = origin
        origin = []
        self.base_vectors_rot = []
        self.base_vectors_quat = []
        self.base = [np.array([1,0,0], dtype=np.float),np.array([0,1,0], dtype=np.float),np.array([0,0,1], dtype=np.float)]
        
        for b in self.base:
            array = np.array(b)
            self.base_vectors_quat.append(quat.rotate(b))
            mat = np.dot(rot_mat, array)
            try:
                self.base_vectors_rot.append(np.array([mat[0,0], mat[0,1], mat[0, 2]]))
            except:
                self.base_vectors_rot.append(np.array([mat[0], mat[1], mat[2]]))
            # print('base vectors')
            # print(self.base_vectors_rot)

    def render_with_bias(self, bot, bias):
        bot.renderer.begin_rendering()
        #Render base coord system
        render_intertial_coordinate_system(bot)
        o = self.origin
        r = bot.renderer
        quat_colors = [r.red(), r.green(), r.blue(), r.black()]
        rot_colors = [r.orange(), r.grey(), r.cyan(), r.black()]

        # Render ux, uy, uz from rot_mat at position   
        for i, v in enumerate(self.base_vectors_quat):
            #scale unit vector
            scaled = v*500
            bot.renderer.draw_line_3d(o, o+scaled, quat_colors[i])
        # b = self.basis_vectors_rot
        # x = b[0][:]
        # y = b[1][:]
        # z = b[2][:]
        # vecs = [x,y,z]
        for i, v in enumerate(self.base_vectors_rot):
            scaled=np.asarray(v)*500
            bot.renderer.draw_line_3d(o+bias, o+scaled+bias, rot_colors[i])
        
        bot.renderer.end_rendering()