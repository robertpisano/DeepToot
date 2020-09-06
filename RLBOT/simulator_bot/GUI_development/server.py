# Server will house a GUI object and run the .main_loop() function

from tkinter import Tk, filedialog, Toplevel, Frame, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar, Checkbutton, Canvas, PhotoImage, NW
from threading import Thread
import socket
import select
import pickle
from DeepToot.RLBOT.simulator_bot.GUI_development.socket_test import TestSocketClass
from rlbot.agents.base_agent import BaseAgent
from DeepToot.RLBOT.simulator_bot.GUI_development.msg_protocol import *

class GUI():
    def __init__(self):
        None
    
    def initialize_frame(self):
        None


class ICPThread(Thread):
    def __init__(self, ip:"", port:int, bot:BaseAgent):
        Thread.__init__(self)
        self.bot = bot

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #Setup TCP Socket
        self.socket.bind((ip, port))
        self.socket.listen(5)
        self.setDaemon(True)
        self.start()

    def run(self):
        """Run the socket "server"
        """        

        while True:
            try:
                client, addr = self.socket.accept()
        
                ready = select.select([client,],[], [],2)
                if ready[0]:
                    print('ready[0] true')
                    received_class = self.recvall(client)
                    # Set bot inside of received class
                    received_class.bot = self.bot 
                    # Get method attribute from process list and run method
                    # TODO: Makepip this a loop that will run through the process list
                    getattr(received_class, received_class.scenario_initialization_process_list[0])()
                    
            except socket.error as error:#, msg:
                print("Socket error! %s" % error.strerror)
                break
            
        # shutdown the socket
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
    
        self.socket.close()
    
    # Get data from socket in loop in case serialization is > buffer size
    def recvall(self, sock):
        BUFF_SIZE = 4096 # 4 KiB
        data = b''
        while True:

            part = sock.recv(BUFF_SIZE)
            data += part
            if len(part) <= 0:
                # either 0 or end of data
                break
            print('receiving buffer from socket length: ' + str(len(part)))

        received_class = pickle.loads(data)
        return received_class       

if __name__ == "__main__":
    icp = ICPThread(SERVER, PORT, BaseAgent(None, None, None))
    icp.run()