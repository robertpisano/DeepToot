from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.MinimumTimeToBallBrain import MinimumTimeToBallBrain


class BrainFactory:
    list = {'MinimumTimeToBall':MinimumTimeToBallBrain}

    @staticmethod
    def create(key):
        return BrainFactory.list[key]()