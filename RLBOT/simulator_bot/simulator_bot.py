from rlbot.agents.base_agent import BaseAgent
from rlbot.utils.structures.game_data_struct import GameTickPacket
from controller_input import controller
from simulator_utilities import GUI
from threading import Thread
from tkinter import Tk

class Agent(BaseAgent):
    def initialize_agent(self):
        # Initialize GUI class, put mainloop() function in thread to it can run concurrently with get_output()
        self.gui = GUI()
        self.gui_thread = Thread(target=self.gui_loop, daemon = False)
        self.gui_thread.start()

    def get_output(self, game_tick_packet):

        # print(self.gui.test_val) # Gui updating test

        return controller.get_output()

    def gui_loop(self):
        try:
            self.gui.__init__() # Re-initialize the GUI for the fuck of it
            while(1): # Continuiously run mainloop() to update data from the gui 
                self.gui.master.mainloop()
        except Exception as e:
            print(e)

# Run this file with F5 in VSCode to run without the overhead of RLbot and the interface to rocket league. Only for debugging
if __name__ == "__main__":
    # Run class to make debugging code
    agent = Agent('name', 'team', 'index')
    agent.initialize_agent()
    while(1):
        agent.get_output(GameTickPacket())
