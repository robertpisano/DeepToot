# Regular import of modules
import DeepToot.src.meta_data_objects.controllers.Controller
import DeepToot.src.meta_data_objects.controllers.AerialController
import DeepToot.src.meta_data_objects.controllers.DrivingController
from DeepToot.src.meta_data_objects.controllers.TPNDrivingController import TPNDrivingController
import DeepToot.src.meta_data_objects.controllers.OpenLoopAerialController
# Reloading of modules for when ControllerFactory is reloaded (A sort of recursive reload)
from importlib import reload
reload(DeepToot.src.meta_data_objects.controllers.Controller)
reload(DeepToot.src.meta_data_objects.controllers.AerialController)
reload(DeepToot.src.meta_data_objects.controllers.DrivingController)
reload(DeepToot.src.meta_data_objects.controllers.TPNDrivingController)
reload(DeepToot.src.meta_data_objects.controllers.OpenLoopAerialController)

from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.controllers.AerialController import AerialController
from DeepToot.src.meta_data_objects.controllers.DrivingController import DrivingController
from DeepToot.src.meta_data_objects.controllers.TPNDrivingController import TPNDrivingController
from DeepToot.src.meta_data_objects.controllers.OpenLoopAerialController import OpenLoopAerialController

from marshmallow import Schema, fields, post_load

class ControllerFactory:
    list = {'AerialController':AerialController, 'DrivingController':DrivingController, 'TPNDrivingController':TPNDrivingController,
            'OpenLoopAerialController':OpenLoopAerialController}

    @staticmethod
    def create(key, **kwargs):
        if key is not "":
            return ControllerFactory.list[key](**kwargs)
        else: 
            return Controller()

class ControllerSchema(Schema):
    name = fields.Str()
    params = fields.Str()
    miscOptions = fields.Str()

    @post_load
    def make_obj(self, data, **kwargs):
        controller = ControllerFactory.create(data['name'])
        controller.params = data['params']
        controller.miscOptions = data['miscOptions']
        return controller