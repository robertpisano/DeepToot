from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject

class AerialController(Controller):
    params = {"kq" : 10, "kw" : 1}
    miscOptions = {"option1":False, "option2":True}

    def __init__(self):
        pass    