from train_module import TrainModule
from time import time
import model_config as c
from learner_model import LearnerModel
from torchvision.models import resnet50

# import model that you want to train. Can be any type of model but make sure the final connected layer is named as 'fc'
def main():
    stime = time()
    tm = TrainModule(LearnerModel())
    # tm = TrainModule(resnet50())
    ftime = time()
    print(f"took {ftime-stime}s to load TrainModule")

    print("Initialising Training ...")
    stime = time()
    tm.train()
    ftime = time()
    print(f"Took {ftime-stime}s to train model, check {c.save_dir} for all saved models")


if __name__ == "__main__":
    main()