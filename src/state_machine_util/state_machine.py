
class StateMachine:
    def __init__(self):
        self.states = {"dwell":"dwell", "set_obj_params":"set_obj_params",\
            "pre_calc":"pre_calc", "init_env":"init_env",\
            "run":"run", "run_save":"run_save", "complete":"complete",\
            "update_objs":"update_objs"}
        self.current_state = self.states["dwell"]

class Dwell():
    @staticmethod
    def do():
        pass
class SetObjParams():
    @staticmethod
    def do():
        pass
class PreCalc(): pass
class InitEnv(): pass
class Run(): pass
class RunSave(): pass
class Complete(): pass
class UpdateObjs(): 
    @staticmethod
    def do():
        pass