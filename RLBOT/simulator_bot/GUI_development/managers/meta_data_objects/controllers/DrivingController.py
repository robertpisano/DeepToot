from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller

class DrivingController(Controller):
    params = {"kp":1, "kd":1, "A":2, "B":3, "C":4}
    miscOptions = {"opt1":None}
    def __init__(self):
        self.__name__ = 'DrivingController'
        pass