from os import F_TEST
import torch.nn as nn
import torch
from torch.nn.modules import flatten


class Alexnet(nn.Module):
    """First neural network to kicks start deeplearning"""
    def __init__(self, num_classes=1000):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size = 11,stride = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride = 2),
            nn.Conv2d(48, 128, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride = 2),
            nn.Conv2d(128, 192, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class MyModule(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, num_classes=10):
        super(MyModule, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,5,kernel_size =3,stride =1),
            nn.ReLU(),
            nn.Conv2d(5,10,kernel_size =3,stride =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1210, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
