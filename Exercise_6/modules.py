import torch
import torch.nn as nn


class Dropout(nn.Module):
    
    def __init__(self, p=0.1):
        super().__init__()
        # store p
        self.p = p
        
    def forward(self, x):
        # In training mode, set each value 
        # independently to 0 with probability p
        # and scale the remaining values 
        # according to the lecture
        # In evaluation mode, return the
        # unmodified input
        if self.training:
            d_m = torch.rand(x.shape)
            d_m = d_m > self.p
            x = x * d_m
            x = x / (1 - self.p)
            return x
        else:
            return x
    