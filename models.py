import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class MetaNet(nn.Module):
    def point_grad_to(self, target):
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            p.grad.data.zero_()
            p.grad.data.add_(p.data - target_p.data)

# https://github.com/vsitzmann/siren

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Siren(MetaNet):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net.append(nn.Tanh())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        return self.net(coords)
    
    def clone(self):
        clone = Siren(self.in_features, self.out_features, self.hidden_features, self.hidden_layers,
                      self.first_omega_0, self.hidden_omega_0)
        clone.load_state_dict(self.state_dict())
        clone.cuda()
        return clone