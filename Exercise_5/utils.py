import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # set the layer's weights as discussed in the lecture
        m.weight.data = torch.randn(m.weight.data.size())*((2/m.in_features)**(1/2))

class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set theta_mu and theta_sigma such that the output of
        # forward initially is zero centered and 
        # normalized to variance 1
        self.theta_mu = nn.Parameter(torch.zeros(num_channels))
        self.theta_sigma = nn.Parameter(torch.ones(num_channels))
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        mean = x.mean(dim=0)
        var = x.var(dim=0)
        
        if self.training:
            # specify behavior at training time
            if self.running_mean is None:
                # set the running stats to stats of x
                self.running_mean = mean
                self.running_var = var

            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                self.running_mean = 0.9 * self.running_mean + 0.1 * mean
                self.running_var = 0.9 * self.running_var + 0.1 * var
            
            x = (x - mean)/((var + self.eps)**(1/2))
            x = (self.theta_sigma * x) + self.theta_mu
            return x
        
        else:
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                x = (x - mean)/((var + self.eps)**(1/2))
                x = (self.theta_sigma * x) + self.theta_mu
                return x

            else:
                # use running stats for normalization
                x = (x - self.running_mean)/((self.running_var + self.eps)**(1/2))
                x = (self.theta_sigma * x) + self.theta_mu
                return x