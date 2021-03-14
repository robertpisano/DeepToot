from multiprocessing import Process
import socket
from DeepToot.src.comms_util.comms_protocol import Message, CommsProtocol, Decoder, Encoder
import traceback

from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.controllers.ControllerFactory import ControllerFactory

class Client:
    def __init__(self): pass

    def send_message(self, type, data):
        """Send a message over the socket

        Args:
            type ([type]): the type of message being sent (must come from the CommsProtocol class)
            data ([type]): the class or data to be sent along with the message
        """        
        try:
            """1. Start the socket 
            2. Connect to the socket
            3. Encode the type and data into Message class
            4. Send message class over socket
            5. print the sent data
            6. shutdown client
            7. close client
            """            
            client = socket.socket(socket.AF_INET,
                                socket.SOCK_STREAM)
            client.connect(CommsProtocol.ADDR)
            # Send data over socket
            send_data = Encoder.encode(type, data)
            client.send(send_data)
            print("Send: ", send_data)

            #Shudwon client
            client.shutdown(socket.SHUT_RDWR)

            client.close()
        except Exception as e:
            print(e)
            traceback.print_exc()

class TestClass:
    def __init__(self):
        self.val = 20

if __name__ == "__main__":
    """Test the use of the Client() class, send execute message and TestClass
    """    
    c = Client()
    c.send_message(CommsProtocol.types['update'], TestClass()) 
    c.send_message(CommsProtocol.types['execute'], 'DrivingController')