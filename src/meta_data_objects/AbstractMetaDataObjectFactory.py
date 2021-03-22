from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.controllers.AerialController import AerialController
from DeepToot.src.meta_data_objects.controllers.DrivingController import DrivingController
from DeepToot.src.meta_data_objects.controllers.TPNDrivingController import TPNDrivingController
from DeepToot.src.meta_data_objects.controllers.OpenLoopAerialController import OpenLoopAerialController
from DeepToot.src.meta_data_objects.brains.MinimumTimeToBallBrain import MinimumTimeToBallBrain
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsTesting import InitialConditionsTesting
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko
from DeepToot.src.meta_data_objects.MetaDataObject import MetaDataObject

class AbstractMetaDataObjectFactory():
    controllerList = {'AerialController':AerialController, 'DrivingController':DrivingController, 'TPNDrivingController':TPNDrivingController}
    
    brainList = {'MinimumTimeToBall':MinimumTimeToBallBrain}

    initialConditionsList = {'InitialConditionsGekko' :InitialConditionsGekko}

    @staticmethod
    def create(key):
        try:
            if key is not "":
                obj = AbstractMetaDataObjectFactory.get_obj(key) 
                return obj
            else:
                return MetaDataObject()
        except:
            raise UserWarning
    
    @staticmethod
    def get_obj(key):
        for d in [AbstractMetaDataObjectFactory.controllerList, AbstractMetaDataObjectFactory.brainList, AbstractMetaDataObjectFactory.initialConditionsList]:
            if key in d:
                return d[key]()