import torch
import torchvision
from torchvision import transforms

class MNIST():
    def __init__(self, batch_size=1):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])
        
        self.train_set = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        
        self.test_set = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False, num_workers=2)