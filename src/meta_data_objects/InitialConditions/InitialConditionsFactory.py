import DeepToot.src.meta_data_objects.InitialConditions.InitialConditions
import DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsTesting
import DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko
from importlib import reload
reload(DeepToot.src.meta_data_objects.InitialConditions.InitialConditions)
reload(DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsTesting)
reload(DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko)
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsTesting import InitialConditionsTesting
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko

from marshmallow import Schema, fields, post_load

class InitialConditionsFactory:
    list = {'Testing' :InitialConditionsTesting,'Attack2' :InitialConditions,'Kickoff1' :InitialConditions, 'InitialConditionsGekko':InitialConditionsGekko}

    @staticmethod
    def create(key):
        if key is not "":
            return InitialConditionsFactory.list[key]()
        else:
            return InitialConditions()

class InitialConditionsSchema(Schema):
    params = fields.Str()
    miscOptions = fields.Str()
    name = fields.Str()

    @post_load
    def make_brain(self, data, **kwargs):
        ic = InitialConditionsFactory.create(data['name'])
        ic.params = data['params']
        ic.miscOptions = data['miscOptions']
        return ic