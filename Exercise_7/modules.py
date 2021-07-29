import torch
import torch.nn as nn


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.tensor([[[-1,0],[1,0]],[[-1,1],[0,0]]],dtype=torch.float).unsqueeze(1)


class Conv2d(nn.Module):
    
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = nn.Parameter(kernel)
        self.padding = ZeroPad2d(padding)
        self.stride = stride
        
    def forward(self, x):
        x = self.padding(x)
        z1,x1,y1 = x.shape
        out_channels,a,b,c = self.kernel.shape
        conv_out_y_dim = int((((y1-c)/self.stride)+1))
        conv_out_x_dim = int((((x1-b)/self.stride)+1))
        conv_out = torch.zeros([out_channels,conv_out_x_dim,conv_out_y_dim])
        for out_channel in range(out_channels):
            for j in range(conv_out_x_dim):
                for i in range(conv_out_y_dim):
                    conv_res =  (x[:,(j*self.stride):b+(j*self.stride),(i*self.stride):c+(i*self.stride)])*(self.kernel[out_channel])
                    conv_out[out_channel][j][i] = torch.sum(conv_res)
        # For input of shape C x H x W
        # implement the convolution of x with self.kernel
        # using self.stride as stride
        # The output is expected to be of size C x H' x W'
        return conv_out


class ZeroPad2d(nn.Module):
    
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        
    def forward(self, x):
        # For input of shape C x H x W
        # return tensor zero padded equally at left, right,
        # top, bottom such that the output is of size
        # C x (H + 2 * self.padding) x (W + 2 * self.padding)
        if self.padding == 0:
            return x
        else :
            z1,x1,y1 = x.shape
            for i in range(self.padding):
                vertical_zeros = torch.zeros(x1,z1).unsqueeze(0)
                tensors = [vertical_zeros.T,x,vertical_zeros.T]
                x = torch.cat(tensors,2)
            for i in range(self.padding):
                Horizontal_zeros = torch.zeros(z1,(y1+2*self.padding)).unsqueeze(1)
                tensors = [Horizontal_zeros,x,Horizontal_zeros]
                x = torch.cat(tensors,1)
        return x
