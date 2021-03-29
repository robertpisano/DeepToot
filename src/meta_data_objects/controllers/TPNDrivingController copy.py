from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick
from rlbot.utils.structures.game_data_struct import GameTickPacket
import numpy as np

class TPNDrivingController(Controller):
    name = 'TPNDrivingController'
    params = {"N":5}
    miscOptions = {"opt1":None}
    def __init__(self):
        self.__name__ = 'TPNDrivingController'
        pass

    def calculate_control(self, packet:GameTickPacket, index) -> SimpleControllerState:
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
        vel = np.array([vx, vy, 0])

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
        Ngb = Rgb/np.linalg.norm(Rbg)
        Ncg = Rcg/np.linalg.norm(Rcg)
        Ngc = Rgc/np.linalg.norm(Rgc)

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

        # Calculate angle between Rcb and Rcg
        Tcbcg = np.arcsin(np.cross(Ncb,Ncg))[2]

        # Calculate angle between Rcb and Rbg
        Tcbg = np.arcsin(np.cross(Ncb, Ncg))

        # Calculate angle between orientation and Ncb
        Toricb = np.arcsin(np.cross(ori, Ncb))[2]

        # Calculate the LOS_rate desired
        Wd = Tcbcg

        # Calculate LOS_rate, W for angular velocity of LOS
        Vr = bvel - vel
        W = np.cross(Rcb, Vr) / np.dot(Rcb,Rcb)

        # Calculate closing velocity of car to ball along LOS
        Vc = np.dot(Rcb/np.linalg.norm(Rcb), vel)
        
        # Calculate acceleration from TPN
        N = self.params["N"]
        a = N * (Wd - W[2]) * Vc

        # Print variables for debug
        # print("Wd:{:0.2f}".format(Wd), " | Tcbcg:{:0.2f}".format(Tcbcg), " | Toricb:{:0.2f}".format(Toricb))
        print(ori, Ppcb)

        # Set throttle
        controller_state.throttle = 1
        if Vc < 1400:
            controller_state.boost = True
        # steer = proj_angle[2] * 10
        
        steer = (N * (W[2]))
            
        # Bang-bang steering calc dependent on a and Vc
        # if a > 0: 
        #     controller_state.steer = 1
        # if a < 0:
        #     controller_state.steer = -1
        # if Vc < 0:
        #     direction = np.sign(np.linalg.norm(np.dot(Rcb, Vc)))
        #     controller_state.steer = direction
        controller_state.steer = np.clip(steer, -1, 1)

        return controller_state
