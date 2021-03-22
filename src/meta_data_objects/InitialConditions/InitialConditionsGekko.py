# from DeepToot.src.gekko_util.gekko_util import *
from rlbot.utils.game_state_util import CarState, BallState
from DeepToot.src.meta_data_objects.MetaDataObject import MetaDataObject
from gekko import GEKKO
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from rlbot.utils.game_state_util import Vector3, Physics, Rotator

class InitialConditionsGekko(InitialConditions):
    params: dict
    miscOptions: dict
    name = 'InitialConditionsGekko'
    def __init__(self):
        self.params = {'sxi':-2000.0, 'syi':-2000.0, 'szi':0.0, 'v_magi':1000.0, 
                        'rolli':0.0, 'pitchi':0.0, 'yawi':-1, 
                        'wxi':0.0, 'wyi':0.0, 'wzi':0.0, 
                        'bxi':-2000.0, 'byi':1500.0, 'bzi':0.0,
                        'bvxi':1000.0, 'bvyi':0.0, 'bvzi':0.0,
                        'broll':0.0, 'bpitch':0.0, 'byaw':0.0,
                        'bwxi':0.0, 'bwyi':0.0, 'bwzi':0.0,
                        'sxf':0.0, 'syf':0.0, 'szf':0.0,
                        'v_magf':0.0, 'vyf':0.0, 'vzf':0.0,
                        'rollf':0.0, 'pitchf':0.0, 'yawf':0.0
                        }
        self.miscOptions = {}
        pass