import time
import numpy as np
import torch
import math

import torch.nn as nn
import torch.nn.functional as F

class FourierFeaturesBlock(nn.Module):
    """
    Maps position-time inputs to Fourier features
    Args:
        L (int): Number of Fourier features / 2
        FS (int): Sample rate of impulse responses
    """
    
    def __init__(self, L, FS):
        super(FourierFeaturesBlock, self).__init__()
        
        self.L = L
        self.stuff = stuff
        self.B = np.array([np.pi*2**l for l in range(L)])
        self.FS = FS
    
    def forward(self, p, t):
        """
        Args:
            p of shape (batch_size, p_dim): positional input
            t of shape (batch_size,): time input
        
        Returns:
            out of shape (batch_size, (p_dim+1)*L*2): fourier feature outputs
        """
#         def fourier_features_mapping(p, L):

            # B = np.repeat([[2**l for l in range(L)]], len(p[0]), axis=0)
            
            print(B.shape)
            x_out = np.empty((len(p),1,2*L*len(p[0])))
            print(x_out.shape)
            for i,x in enumerate(p):
                x_proj = np.pi*x[:,np.newaxis]*B[np.newaxis,:]
                print(x_proj.shape)
                x_out[i] = np.reshape(np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1), (1,-1))
            return x_out
        
        proj = x * self.B