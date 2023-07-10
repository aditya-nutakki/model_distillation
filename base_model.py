# import torch
# from torchvision.models import resnet50

# base_model = resnet50().to(device="cuda")
# print(base_model)

# ----------------------
import urllib.request
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

import torch

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

input_image = Image.open(filename)

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(input_image).unsqueeze(dim=0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
