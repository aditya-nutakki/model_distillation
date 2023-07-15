import os
from train_module import TrainModule
from learner_model import LearnerModel
from time import time


def main():
    # Load a model. This should have the weights loaded. This will be taken care of based on params given in model_config.py Make sure to set mode to "infer"
    stime = time()
    tm = TrainModule(LearnerModel())
    images_to_infer = [] # list of image paths
    
    stime = time()
    tm.infer(images_to_infer)
    ftime = time()
    print(f"Took {ftime-stime}s to infer {len(images_to_infer)} images")

if __name__ == "__main__":
    main()