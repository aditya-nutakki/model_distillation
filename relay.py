# import torch
# from learner_model import LearnerModel
# # from base_model import resnet50
# from train_module import TrainModule
# from torchvision.models import resnet50
# import time


# tm = TrainModule(LearnerModel())
# _resnet = resnet50(pretrained=False)
# tm.learn_from(_resnet)

# stime = time.time()
# tm = TrainModule(LearnerModel())
# ftime = time.time()
# print(f"Took {ftime-stime}s to load module")

# stime = time.time()
# tm.infer(["/opt/infilect/aditya/datasets/rice_image_dataset/splits/test/Jasmine/Jasmine (9902).jpg", 
#           "/opt/infilect/aditya/datasets/rice_image_dataset/splits/test/Arborio/Arborio (9922).jpg"])
# ftime = time.time()
# print(f"took {ftime-stime}s")



# _resnet = resnet50(pretrained=False)
# tm.learn_from(_resnet)
# # print(_resnet)
# tm = TrainModule(_resnet)
# tm.train()




