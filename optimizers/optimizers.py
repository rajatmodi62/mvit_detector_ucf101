#contains all the losses which are used by the 
#returns : the loss object 


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import slowfast.models.optimizer as optim
import torch.optim as optim
#make this dependent  on cfg later on 
def build_optimizer(model,cfg):
        #optimizer = optim.construct_optimizer(model, cfg)
        optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=[0.5, 0.999], weight_decay=0, eps=1e-6)
        return optimizer
