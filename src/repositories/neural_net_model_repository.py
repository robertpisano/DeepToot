# neural net models (for now) will be saved as files in our codebase
# the retrieval of these models is something that will be based on a given strategy - which requires logic for finding a file based on a strategy
# NB: we also may wind up needing to use a database or other storage faacility
from  keras.utils import custom_object_scope
from  keras.models import load_model, save_model
from DeepToot.src.data_generation.entities.neural_net.base_neural_net_model import BaseNeuralNetModel
import DeepToot
import os

class NeuralNetModelRepository():
    MODELS_PATH = os.path.dirname(DeepToot.__file__) + "\SavedModels\\"
    MODEL_EXT = ".json"

    def __init__(self, model: BaseNeuralNetModel):
        self.model = model

    def _get_file_path(self):
        #model name to use for file name
        model_type = self._model_type()
        #create model directory if it doesn't exist
        if not os.path.exists(self.MODELS_PATH): os.mkdir(self.MODELS_PATH)
        #create file name
        return self.MODELS_PATH + model_type

    def _model_type(self):
        return self.model.__class__.__name__

    def save(self):
        file_path = self._get_file_path()
        model_type = self._model_type
        print("saving model {model_type} to {file_name}".format(model_type=model_type, file_name=file_path))
        self.model.save(file_path, overwrite=True)

    def load(self):
        """Use self.arch.file_name to grab file
        """
        return load_model(self._get_file_path())

if __name__ == "__main__":
    model_type = "poop"
    MODEL_EXT = ".json"
    print("os.path.dirname(DeepToot.__file__)")
    print(os.path.dirname(DeepToot.__file__))
    MODELS_PATH = os.path.dirname(DeepToot.__file__) + "\SavedModels\\"
    if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)
    file_name =  MODELS_PATH + model_type + MODEL_EXT
    print("saving model as file")
    print(file_name)