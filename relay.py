import torch
from learner_model import LearnerModel
# from base_model import resnet50
from train_module import TrainModule

tm = TrainModule(LearnerModel())
tm._train()



