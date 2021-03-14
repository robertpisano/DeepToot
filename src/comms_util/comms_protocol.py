import pickle

class CommsProtocol:
    types = {"execute": "execute", "update": "update", "none": "None"}
    som = "%%%" # Start of message key
    sod = "$$$" # Start of data key
    eom = "###" # End of message key
    SERVER = '127.0.0.1'
    PORT = 5050
    ADDR = (SERVER, PORT)
    FORMAT = 'utf-8'

class Message:
    def __init__(self, type = "None", data = None):
        self.type = type
        self.data = data

class Decoder:
    @staticmethod
    def decode(msg: str):
        """Convert the message string into it's message type and data parts

        Args:
            msg (Message): [description]
        """        
        # Split by the som, sod, eod
        msg_type, data = msg.split(bytes(CommsProtocol.sod, 'utf-8'))
        msg_type = msg_type.decode('utf-8').replace(CommsProtocol.som, '') # Remove som
        data = str(data).replace(CommsProtocol.eom, '') # Remove eom
        return Message(msg_type, data)

class Encoder:
    @staticmethod
    def encode(msg_type, data):
        """Encode the message into bytes using som, sod, eod from CommsProtocol

        Args:
            type ([type]): [description]
            data ([type]): [description]
        """        
        string = bytes(CommsProtocol.som, 'utf-8')  # Add start of msg (som) 
        string += bytes(msg_type, 'utf-8')              # Add type string
        string += bytes(CommsProtocol.sod, 'utf-8') # Add start of data (sod)
        string += pickle.dumps(data)                # Add data
        string += bytes(CommsProtocol.eom, 'utf-8') # Add end of msg (eom)
        return string
