import torch
import torchvision

from torchvision.datasets import CIFAR10
from torchvision import transforms
cifar = CIFAR10(transform=pipeline)
pipeline = transforms.Compose(transforms.ToTensor)
from torch.utils.data import DataLoader
loader = DataLoader(cifar, batch_size=16, shuffle=True)

class netmodel(torch.nn.Module):
    def __init__(self):
        super.__init__()
        self.layer1 = torch.nn.Linear(16,32)
        self.activation1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(32,32)
        self.activation2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(1024,2)

    def forward(self, input):
        buffer = self.layer1(input)
        buffer = self.activation1(buffer)
        buffer = self.layer2(buffer)
        buffer = self.activation2(buffer)
        buffer = buffer.flatten(start_dim = 1)
        return self.out(buffer)
from torchvision.datasets import MNIST

MNIST()

dataloader = torch.utils.DataLoader(dataset, batch_size=1, suffle=False)
loss = torch.nn.MSELoss()
optim = torch.optim.SGD()

