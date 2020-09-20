from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.AerialController import AerialController
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.DrivingController import DrivingController
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.OpenLoopAerialController import OpenLoopAerialController
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.MinimumTimeToBallBrain import MinimumTimeToBallBrain
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditionsTesting import InitialConditionsTesting
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject

class AbstractMetaDataObjectFactory():
    list = {'AerialController':AerialController, 'Driving Controller':DrivingController,
    'MinimumTimeToBall':MinimumTimeToBallBrain,
    'Testing' :InitialConditionsTesting,'Attack2' :InitialConditions,'Kickoff1' :InitialConditions}

    @staticmethod
    def create(key):
        if key is not "":
            return AbstractMetaDataObjectFactory.list[key]()
        else:
            return MetaDataObject()

