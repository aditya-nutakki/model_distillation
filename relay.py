import torch
from learner_model import LearnerModel
# from base_model import resnet50
from train_module import TrainModule
from torchvision.models import resnet50

# tm = TrainModule(LearnerModel())
# tm._train()


_resnet = resnet50(pretrained=False)
# print(_resnet)
tm = TrainModule(_resnet)
tm._train()




