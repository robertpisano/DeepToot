import numpy as np

def convert_from_euler_angles(roll, pitch, yaw):

    CR = np.cos(roll)
    SR = np.sin(roll)
    CP = np.cos(pitch)
    SP = np.sin(pitch)
    CY = np.cos(yaw)
    SY = np.sin(yaw)

    theta = np.zeros((3,3))

    # front direction
    theta[0, 0] = CP * CY
    theta[1, 0] = CP * SY
    theta[2, 0] = SP

    # left direction
    theta[0, 1] = CY * SP * SR - CR * SY
    theta[1, 1] = SY * SP * SR + CR * CY
    theta[2, 1] = -CP * SR

    # up direction
    theta[0, 2] = -CR * CY * SP - SR * SY
    theta[1, 2] = -CR * SY * SP + SR * CY
    theta[2, 2] = CP * CR

    return theta