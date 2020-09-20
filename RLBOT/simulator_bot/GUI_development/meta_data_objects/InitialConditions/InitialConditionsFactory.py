from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditionsTesting import InitialConditionsTesting

class InitialConditionsFactory:
    list = {'Testing' :InitialConditionsTesting,'Attack2' :InitialConditions,'Kickoff1' :InitialConditions}

    @staticmethod
    def create(key):
        if key is not "":
            return InitialConditionsFactory.list[key]()
        else:
            return InitialConditions()