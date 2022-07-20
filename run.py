import argparse
from argparse import ArgumentParser
from models import load_model
import json
import os

if __name__ == '__main__':
    #Initializing with given configurations
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    args = argparse.Namespace(**hparam)

    #Setting up GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES
    
    #Load and use model for inference
    model = load_model(args.model)
    model = model(args)
    model.get_dataset()
    model.evaluate()