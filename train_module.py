import os
import model_config as c
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import datasets, transforms
import time
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

class TrainModule():
    def __init__(self, model) -> None:
        self.device = c.device
        # print(model, type(model))
        self.model = model.to(self.device) # expecting a torch.nn.Module
        self.num_epochs = c.epochs
        self.batch_size = c.batch_size
        self.input_dims = c.input_dims
        self.param_count = self.get_total_param_count(self.model)

        print(f"Loaded Model with {self.param_count} total parameters !")

        self.opt = SGD(self.model.parameters(), lr=3e-4)
        self.loss_fn = CrossEntropyLoss()

        if c.save_model_name:
            self.save_dir = os.path.join(c.save_dir, c.save_model_name)
        else:
            self.save_dir = None

        os.makedirs(self.save_dir, exist_ok=True)

        # self.train_path, self.test_path = c.train_path, c.test_path
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.input_dims[1], self.input_dims[2])), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.train_set, self.test_set = datasets.ImageFolder(c.train_path, transform=self.transforms), datasets.ImageFolder(c.test_path, transform=self.transforms)
        self.train_loader, self.test_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True), DataLoader(self.test_set, batch_size=self.batch_size)


    def get_total_param_count(self, model):
        return sum(p.numel() for p in model.parameters())


    def _train(self):
        torch.cuda.empty_cache()
        print(f"Starting training !")

        for _e in range(self.num_epochs):
            print(f"epoch {_e+1}:")
            stime = time.time()
            running_loss, last_loss = 0.0, 0.0
            
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.opt.zero_grad()

                preds = self.model(images)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                # print(loss)
                self.opt.step()

                running_loss += loss.item()

                if i%100 == 0:
                    last_loss = running_loss/100
                    print(f"batch loss = {last_loss}")
                    running_loss = 0.0


            ftime = time.time()
            print()
            print(f"Took {ftime-stime}s to complete epoch number {_e}")
            if self.save_dir:
                torch.save(self.model.state_dict(), self.save_dir + f"_epoch{_e}")
                

            print()

        

