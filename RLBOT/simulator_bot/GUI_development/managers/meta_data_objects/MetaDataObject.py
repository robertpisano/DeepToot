class MetaDataObject():
    params: dict
    miscOptions: dict
    __name__ = 'MetaDataObject'

    def __init__(self):
        pass

    # unused! was trying something
    def init_from_dict(self, dict, name):
        self.params = dict[name].params