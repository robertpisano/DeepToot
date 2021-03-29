from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick
from rlbot.utils.structures.game_data_struct import GameTickPacket
import numpy as np
from scipy.interpolate import CubicSpline

class TPNDrivingController(Controller):
    name = 'TPNDrivingController'
    params = {"N":5}
    miscOptions = {"opt1":None}
    def __init__(self):
        self.__name__ = 'TPNDrivingController'
        cur = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_cur = np.array([0,500,1000,1500,1750,2300])
        v_cur_fine = np.linspace(0,2300,100)
        cur_fine = np.interp(v_cur_fine, v_cur, cur)
        self.curvature = CubicSpline(v_cur_fine, cur_fine)
        pass

    def calculate_control(self, packet:GameTickPacket, index) -> SimpleControllerState:
        # params as local variables
        N = self.params["N"]

        # Goal position
        gpos = np.array([0,5120,0])

        # define controller_state for return
        controller_state = SimpleControllerState()

        # Get car state variables
        car = packet.game_cars[index].physics
        x = car.location.x
        y = car.location.y
        vx = car.velocity.x
        vy = car.velocity.y
        yaw = car.rotation.yaw
        ori = np.array([np.cos(yaw), np.sin(yaw), 0])
        cpos = np.array([x, y, 0])
        cvel = np.array([vx, vy, 0])

        # Get ball state variables
        ball = packet.game_ball.physics
        bx = ball.location.x
        by = ball.location.y
        bvx = ball.velocity.x
        bvy = ball.velocity.y
        bpos = np.array([bx, by, 0])
        bvel = np.array([bvx, bvy, 0])

        # Calculate Line of sights (LOS), Rcb, Rcg, Rbg
        Rcb = bpos - cpos
        Rbc = cpos - bpos
        Rbg = gpos - bpos
        Rgb = bpos - gpos
        Rcg = gpos - cpos
        Rgc = cpos - gpos


        
        # Calculate the LOS norms
        Ncb = Rcb/np.linalg.norm(Rcb)
        Nbc = Rbc/np.linalg.norm(Rbc)
        Nbg = Rbg/np.linalg.norm(Rbg)
        Ngb = Rgb/np.linalg.norm(Rgb)
        Ncg = Rcg/np.linalg.norm(Rcg)
        Ngc = Rgc/np.linalg.norm(Rgc)

        # Calculate relative velocities
        Vcb = bvel - cvel
        Vc = -1*np.dot(Vcb, Ncb)

        # Calculate projection of Rcg onto Ngb
        Pcgbg = ((np.dot(Rgc, Ngb)/np.dot(Ngb, Ngb))*Ngb) - Rgc
        Ng = Pcgbg/np.linalg.norm(Pcgbg)
        proj_angle = np.arcsin(np.cross(ori, Ng))

        # Calculate projection of projection onto Rcb
        Ppcb = ((np.dot(Pcgbg, Rcb)/np.dot(Rcb, Rcb)) * Rcb)

        # Get desired angle
        proj_sum = Pcgbg + Ppcb
        normal = proj_sum / np.linalg.norm(proj_sum)
        proj_angle = np.arcsin(np.cross(ori, normal))

        #calculate angle between Rcb and Rcg
        Acbg = self.get_angle(Ncg, Ncb)
        # Calculate cross between Rcb and Rcg
        Tcbcg = np.cross(Ncg, Ncb)
        # Calculate the LOS_rate desired
        Wd = Tcbcg
        # Calculate desired velcoity vector from LOS_rate desired
        Vlos = np.cross(Wd, Rcb)
        # Calculte desired closing velocity vector dependent on angle between Rcb and Rcg
        Vc_mag = np.clip(1000+1000 * ((np.pi/2) - Acbg) / (np.pi/2), 0,1000)
        Vcd = Vc_mag * Ncb
        
        # Calculate total desired velocty by combining Vlos and Vcd
        V_total = Vcd + Vlos

        # Set throttle
        controller_state.throttle = 1
        if Vc < 2000 and np.abs(Acbg) < 0.5:
            controller_state.boost = True
        steer = proj_angle[2] * 10
        
        # Calculate steering input based on angle between desired total velocity velocity and orientation
        Vtn = V_total/np.linalg.norm(V_total)
        error = self.get_angle(ori, Vtn)
        Vlos_n = Vlos/np.linalg.norm(Vlos)
        Vlos_angle = self.get_angle(ori, Vlos_n)
        # steer = (N * (W[2]))
        steer=error*10
        # steer=1
        
        controller_state.steer = np.clip(steer, -1, 1)

        # Print variables for debug
        # print("Wd:{:0.2f}".format(Wd), " | Tcbcg:{:0.2f}".format(Tcbcg), " | Toricb:{:0.2f}".format(Toricb))
        # print(ori, Vtn, error)
        # print(Ncg, Ncb, self.get_angle(Ncg, Ncb))
        # print(self.get_angle(Ncg, Ncb))
        # print(self.determine_maximum_curvature(cvel))
        print(Vc, Acbg)
        return controller_state

    @staticmethod
    def get_angle(vec1, vec2):
        inner = np.dot(vec1, vec2)
        cross = np.cross(vec1,vec2)
        norm = np.linalg.norm(cross)
        if(norm < 1e-3): return 0.0
        cross_norm = cross/norm
        angle = np.arctan2(cross_norm[2], inner)
        return angle

    def determine_maximum_curvature(self, vel):
        vmag = np.linalg.norm(vel)
        return self.curvature(vmag)