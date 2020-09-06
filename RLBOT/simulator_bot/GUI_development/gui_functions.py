from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject

def print_hello():
    print("hello")

def populate_params_table(obj: MetaDataObject):
    if(obj is not None):
        return obj.params
    
    pass

def populate_misc_table(obj: MetaDataObject):
    if(obj is not None):
        return obj.miscOptions
    
    pass