import DeepToot.src.meta_data_objects.brains.MinimumTimeToBallBrain
from importlib import reload
reload(DeepToot.src.meta_data_objects.brains.MinimumTimeToBallBrain)
from DeepToot.src.meta_data_objects.brains.MinimumTimeToBallBrain import MinimumTimeToBallBrain

from marshmallow import Schema, fields, post_load

class BrainFactory:
    list = {'MinimumTimeToBall':MinimumTimeToBallBrain}

    @staticmethod
    def create(key):
        return BrainFactory.list[key]()

class BrainSchema(Schema):
    params = fields.Str()
    miscOptions = fields.Str()
    name = fields.Str()

    @post_load
    def make_obj(self, data, **kwargs):
        b = BrainFactory.create(data['name'])
        b.params = data['params']
        b.miscOptions = data['miscOptions']
        return b