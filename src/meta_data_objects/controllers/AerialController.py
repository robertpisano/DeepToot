from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.MetaDataObject import MetaDataObject

class AerialController(Controller):
    params = {"kq" : 10, "kw" : 1}
    miscOptions = {"option1":False, "option2":True}
    name = 'AerialController'

    def __init__(self):
        pass    