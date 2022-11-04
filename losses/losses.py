#original authors: kevin & ayush 
# just pasted it here 
#contains all the losses which are used by the trainer 
#returns : a dict <loss_name: fwd pass>

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return (1 - dice)


class IoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU

class BCELoss(nn.Module):

    def __init__(self,thresh = 0.5):
        super(BCELoss, self).__init__()
        self.thresh = thresh
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets,):
        
        thresh = self.thresh
        inputs = inputs.float()
        targets = (targets>=thresh)*1
        targets = targets.float()
        #targets = ((targets>=self.thresh)).float()
        # print(inputs.shape, targets.shape, torch.unique(targets), inputs.dtype, targets.dtype)
        loss = self.criterion(inputs, targets)
        
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    
    def forward(self, inputs, targets,):
        # print("classification loss", inputs.shape, targets.shape,inputs.dtype, targets.dtype)
        # print("target ", targets)
        # inputs = inputs.to(torch.float64)
        # targets = targets.to(torch.float64)
        # targets = targets.long()
        #targets = ((targets>=self.thresh)).float()
        # print(inputs.shape, targets.shape, torch.unique(targets), inputs.dtype, targets.dtype)
        loss = self.criterion(inputs, targets)
        return loss
    
#To Do: more implementation of cfg later on 
def build_losses(cfg = None):
    #create loss objects 
   
    loss_dict = {
        'dice_loss': DiceLoss(),\
        'bce_mask_loss': BCELoss(),\
        'cls_loss': ClassificationLoss(),\
            }
    print("reutrning loss_dict")
    return loss_dict


#rmodi:debug loss
# loss = nn.BCEWithLogitsLoss()
# input = torch.randn((1,16,224,224)).float()*2000
# target = ((torch.zeros((1,16,224,224)) > 0.5)).float()
# output = loss(input, target)
# print(input.dtype, target.dtype)


# loss = BCELoss()
# input = torch.randn((1,16,224,224))#.float()*2000
# target = torch.randn((1,16,224,224))
# output = loss(input, target)
# print(input.dtype, target.dtype)

loss = ClassificationLoss()
input = torch.randn((1,25))
target = torch.Tensor([8])
target = target.to(torch.int64)
print("drype", target.dtype)
input = input.float()
# target = target.float()
loss(input, target)
