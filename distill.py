import os
from learner_model import LearnerModel
from train_module import TrainModule
from torchvision.models import resnet50
from time import time

def main():
    # mode in model_config.py should be 'train'
    # make sure 'base_model_path' in model_config.py is the base_models weights
    tm = TrainModule(LearnerModel())
    _resnet = resnet50(pretrained=False)
    tm.learn_from(_resnet)
    # print(_resnet)
    

if __name__ == "__main__":
    main()
