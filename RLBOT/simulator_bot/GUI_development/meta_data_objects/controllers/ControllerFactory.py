from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.AerialController import AerialController
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.DrivingController import DrivingController
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.OpenLoopAerialController import OpenLoopAerialController

class ControllerFactory:
    controllerList = {'AerialController':AerialController, 'Driving Controller':DrivingController}

    @staticmethod
    def create(key):
        if key is not "":
            return ControllerFactory.controllerList[key]()
        else: 
            return Controller()