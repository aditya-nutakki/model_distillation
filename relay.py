import torch
from learner_model import LearnerModel
# from base_model import resnet50
from train_module import TrainModule
from torchvision.models import resnet50
import time


tm = TrainModule(LearnerModel())
# tm._train()

stime = time.time()
tm.infer(["/opt/infilect/aditya/datasets/rice_image_dataset/splits/test/Jasmine/Jasmine (9959).jpg", 
          "/opt/infilect/aditya/datasets/rice_image_dataset/splits/test/Arborio/Arborio (9998).jpg"])
ftime = time.time()
print(f"took {ftime-stime}s")
# _resnet = resnet50(pretrained=False)
# # print(_resnet)
# tm = TrainModule(_resnet)
# tm._train()




