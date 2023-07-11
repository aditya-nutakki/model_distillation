from os import listdir

batch_size = 32
epochs = 30
input_dims = (3, 224, 224) # make sure to modify the beginning and the last layers of any model in train_module to get the correct shapes
# dataset_path = "/mnt/d/work/datasets/rice_image_dataset" # path to folder containing 'train' and 'test'
train_path, test_path = "/mnt/d/work/datasets/rice_image_dataset/train", "/mnt/d/work/datasets/rice_image_dataset/test"
device = "cuda" # or "cpu"
save_dir, model_name = "./models", "example" # .pt extension will be added, 'None' if you dont want to save 
nc = len(listdir(train_path)) # number of classes
