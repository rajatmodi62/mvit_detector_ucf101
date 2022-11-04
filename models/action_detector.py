#author:rmodi
import numpy as np
import os
import pickle
import torch
import torch.nn as nn 
import torch.nn.functional as F
import argparse
from configs.defaults import get_cfg
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu


#My own wrapper containing the slowfast models
def build_rajats_model(cfg):
    model = ActionDetectionModel(cfg)
    return nn.DataParallel(model)
    # return ActionDetectionModel(cfg)


class ActionDetectionModel(nn.Module):

    def __init__(self,cfg):
        super(ActionDetectionModel, self).__init__()
        self.backbone = build_model(cfg)
        if cfg.TRAIN.ENABLE:
            cu.load_checkpoint(
                path_to_checkpoint = cfg.MVIT.CHECKPOINT_PATH,
                model = self.backbone,
                data_parallel=False,)
            print("loaded the model")
        else:
            print("test mode.... backbone weights wont be loaded")
        
        print("backbone init")
        # self.upsample1 = nn.ConvTranspose3d(576,288,kernel_size = (1,3,3), stride = (1,2,2),padding = (0,1,1),output_padding = (0,1,1))
        # self.upsample2 = nn.ConvTranspose3d(576,144,kernel_size = (1,3,3), stride = (1,2,2),padding = (0,1,1),output_padding = (0,1,1))
        # self.upsample3 = nn.ConvTranspose3d(288,144,kernel_size = (1,3,3), stride = (1,2,2),padding = (0,1,1),output_padding = (0,1,1))
        # #LAST BLOCK WITH NO SKIPS 
        # self.upsample4 = nn.ConvTranspose3d(144,1,kernel_size = (3,3,3), stride = (2,2,2),padding = (1,1,1),output_padding = (1,1,1))
        self.upsample1 = nn.ConvTranspose3d(576,288,kernel_size = (3,3,3), stride = (2,2,2),padding = (1,1,1),output_padding = (1,1,1))
        self.upsample2 = nn.ConvTranspose3d(288,144,kernel_size = (1,3,3), stride = (1,2,2),padding = (0,1,1),output_padding = (0,1,1))
        self.upsample3 = nn.ConvTranspose3d(144,72,kernel_size = (1,3,3), stride = (1,2,2),padding = (0,1,1),output_padding = (0,1,1))
        #LAST BLOCK WITH NO SKIPS 
        self.upsample4 = nn.ConvTranspose3d(72,1,kernel_size = (1,3,3), stride = (1,2,2),padding = (0,1,1),output_padding = (0,1,1))
    
    def forward(self, x):
        # print("model input",x.shape)
        # b,c,t,h,w = x.shape
        # seg = torch.zeros((b,t,h,w)).to(x.device)

        x,bbn_feat = self.backbone([x])
        # print("returning", x.shape, seg.shape)
        b1,b2,b3 = bbn_feat

        #print("shhapes",b1.shape, b2.shape, b3.shape)
        # mask = self.upsample1(b3)
        # mask = torch.cat([mask,b2],1)
        # mask = self.upsample2(mask)
        # mask = torch.cat([mask,b1],1)
        # mask = self.upsample3(mask)
        # mask = self.upsample4(mask)
        # mask = mask.squeeze(1) #squeeze channel dimsnions
        # print("mask shape",x.shape, mask.shape)
        # exit(1)

        mask = self.upsample1(b3)
        mask = self.upsample2(mask)
        mask = self.upsample3(mask)
        mask = self.upsample4(mask)
        mask = mask.squeeze(1)
        return x ,mask


        