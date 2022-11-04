#rmoid: responsible for training the model 


#author:rmodi
import numpy as np
import os
import pickle
import torch
import torch.nn as nn 
import argparse
from utils.meters import AverageMeter
from models.model_interface import model_interface
import wandb
from dataloaders.ucf_dataloader import build_dataloader


#train the  model 
def train_single_epoch(cfg,epoch,model,optimizer,losses,logger,device):
    
    print("start train epoch:", epoch)
    model = model.train()
    
    train_loader, val_loader,test_loader,= build_dataloader(cfg)

    dataloader = train_loader
    meters = {
    'dice_meter':AverageMeter(),\
    'bce_meter':AverageMeter(),\
    'cls_meter':AverageMeter(),\
    }
    
    
    for index, data in enumerate(dataloader):
        #define meters 
        
        
        pred_logits, pred_seg_mask,total_loss,curr_loss_dict = model_interface(model, data, losses, device)
        
        total_loss.backward()
        optimizer.step()
        
        bs = pred_logits.shape[0] 
        #update the meteres
        cls_loss = curr_loss_dict['cls_loss']
        dice_loss = curr_loss_dict['dice_loss']
        bce_mask_loss = curr_loss_dict['bce_mask_loss']
        meters['cls_meter'].update(cls_loss,bs)
        meters['dice_meter'].update(dice_loss,bs)
        meters['bce_meter'].update(bce_mask_loss,bs)

        if index%cfg.PRINT_FREQ ==0:
            print("Done Epoch {}/{} {}/{} Class Loss:{:.2f}, Dice Loss:{:.3f} Bce Loss:{:.3f}".format(
                epoch+1, 
                cfg.NUM_EPOCHS,
                index+1,
                len(dataloader),
                meters['cls_meter'].avg, 
                meters['dice_meter'].avg, 
                meters['bce_meter'].avg
            ))
        wandb.log({"progress/train/epoch": epoch})
        wandb.log({"progress/train/idx": index})
        wandb.log({"train/cls_loss": meters['cls_meter'].avg})
        wandb.log({"train/dice_loss": meters['dice_meter'].avg})
        wandb.log({"train/bce_mask_loss": meters['bce_meter'].avg})
        # break
    #reset thhe meters 
    meters['cls_meter'].reset()
    meters['dice_meter'].reset()
    meters['bce_meter'].reset()
    total_loss= 0
    for k,meter in meters.items():
        if meter!='bce_meter': #dont apply bce loss
            total_loss+=meter.avg
    return total_loss


def test_single_epoch(cfg,epoch,model,optimizer,losses,logger,device):
    model = model.eval()
    print("start test epoch:", epoch)
    train_loader, val_loader,test_loader,= build_dataloader(cfg)

    dataloader = val_loader
    meters = {
    'dice_meter':AverageMeter(),\
    'bce_meter':AverageMeter(),\
    'cls_meter':AverageMeter(),\
    }
    for index, data in enumerate(dataloader):
        #define meters 
        
        with torch.no_grad():
            pred_logits, pred_seg_mask,total_loss,curr_loss_dict = model_interface(model, data, losses, device)
        
        # total_loss.backward()
        # optimizer.step()
        
        bs = pred_logits.shape[0] 
        #update the meteres
        cls_loss = curr_loss_dict['cls_loss']
        dice_loss = curr_loss_dict['dice_loss']
        bce_mask_loss = curr_loss_dict['bce_mask_loss']
        meters['cls_meter'].update(cls_loss,bs)
        meters['dice_meter'].update(dice_loss,bs)
        meters['bce_meter'].update(bce_mask_loss,bs)

        if index%cfg.PRINT_FREQ ==0:
            print("Done Epoch {}/{} {}/{} Class Loss:{:.2f}, Dice Loss:{:.3f} Bce Loss:{:.3f}".format(
                epoch+1, 
                cfg.NUM_EPOCHS,
                index+1,
                len(dataloader),
                meters['cls_meter'].avg, 
                meters['dice_meter'].avg, 
                meters['bce_meter'].avg
            ))
        wandb.log({"progress/test/epoch": epoch})
        wandb.log({"progress/test/idx": index})
        wandb.log({"test/cls_loss": meters['cls_meter'].avg})
        wandb.log({"test/dice_loss": meters['dice_meter'].avg})
        wandb.log({"test/bce_mask_loss": meters['bce_meter'].avg})
        # break
    
    meters['cls_meter'].reset()
    meters['dice_meter'].reset()
    meters['bce_meter'].reset()
    total_loss= 0
    for k,meter in meters.items():
        total_loss+=meter.avg
    return total_loss