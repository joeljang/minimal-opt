from models.OPT import OPT as OPT

def load_model(type: str):
    if type=='opt':
        return OPT
    else:
        raise Exception('Select the correct model type. Only supporting OPT now.')