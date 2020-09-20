from rlbot.agents.base_agent import BaseAgent
from controller_input import controller

import pandas as pd
import os
import DeepToot


class Agent(BaseAgent):



    def initialize_agent(self):
        # This runs once before the bot starts up
        # self.controller_state = SimpleControllerState()
        # self.nn_manager = NeuralNetworkManager()
        # self.sc = ScenarioCreator()
        # self.sc.hardcoded_load()

        col = ["time", "location_x", "location_y", "location_z",
            "velocity_x", "velocity_y", "velocity_z", 
            "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
            "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "controller_throttle", "controller_steer", "controller_boost",
            "ping_flag"]
        self.dataframe = pd.DataFrame([], columns = col)
        self.ping_flag = 0


    def get_output(self, game_tick_packet):
        p = game_tick_packet.game_cars[self.index].physics
        controller_output = controller.get_output()  

        if(controller_output.throttle > 0):
            self.ping_flag = 1

        else:
            self.ping_flag = 0

        quat = self.get_rigid_body_tick().players[self.index].state.rotation

        data = [game_tick_packet.game_info.seconds_elapsed, p.location.x, p.location.y, p.location.z, 
                p.velocity.x, p.velocity.y, p.velocity.z, 
                quat.w, quat.x, quat.y, quat.z,
                p.angular_velocity.x, p.angular_velocity.y, p.angular_velocity.z,
                controller_output.throttle, controller_output.steer, controller_output.boost, 
                self.ping_flag]

        a_series = pd.Series(data, index=self.dataframe.columns)
        self.dataframe = self.dataframe.append(a_series, ignore_index=True)

        # try:
        #     user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
        #     print(user_paths)
        # except KeyError:
        #     user_paths = []

        if(controller_output.save_data):
            data_path = os.path.dirname(DeepToot.__file__) + "\SavedData\stupid.csv"
            self.dataframe.to_csv(data_path)
            print("saved_training_data_betch")


        return controller_output
    
    def _get_file_path(self):
            #model name to use for file name
        model_type = self._model_type()
        #create model directory if it doesn't exist
        if not os.path.exists(self.MODELS_PATH): os.mkdir(self.MODELS_PATH)
        #create file name
        return self.MODELS_PATH + model_type
    
if __name__ == "__main__":
    a = Agent([],[],[])