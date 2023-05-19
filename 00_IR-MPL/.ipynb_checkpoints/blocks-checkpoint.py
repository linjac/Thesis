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
    """
    
    def __init__(self, L):
        super(FourierFeaturesBlock, self).__init__()
        
        self.L = L
        self.stuff = stuff
        
    def forward(self, p, t):
        """
        Args:
            p of shape (batch_size, p_dim): positional input
            t of shape (batch_size,): time input
        
        Returns:
            out of shape (batch_size,): fourier feature outputs
        """
        