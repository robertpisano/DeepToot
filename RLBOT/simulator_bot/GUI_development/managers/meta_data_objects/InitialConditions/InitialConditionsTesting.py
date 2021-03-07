from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from pyquaternion import Quaternion

class InitialConditionsTesting(InitialConditions):
    params = {'carPosition':(0.0, 0.0, 0.0), 'carVelocity':(0.0, 0.0, 0.0),
               'carOrientation':(0.0, 0.0, 0.0, 0.0), 'carAngVel':(0.0, 0.0, 0.0),
               'ballPosition':(0.0, 0.0, 0.0), 'ballVelocity':(0.0, 0.0, 0.0),
               'ballOrientation':(0.0, 0.0, 0.0, 0.0), 'ballAngVel':(0.0, 0.0, 0.0)}
    
    miscOptions = {}
    

    