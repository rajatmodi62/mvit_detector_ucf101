#author:rmodi
import numpy as np
import os
import pickle
import torch
import torch.nn as nn 
import argparse
from configs.defaults import get_cfg
# from slowfast.models import build_model
import slowfast.utils.checkpoint as cu
from models.action_detector import build_rajats_model
from models.model_interface import model_interface
from optimizers.optimizers import build_optimizer
from utils.rajat_logger import build_logger
from dataloaders.ucf_dataloader import build_dataloader
from losses.losses import build_losses
from utils.meters import AverageMeter
import wandb
from trainer.trainer import train_single_epoch, test_single_epoch
import random, os
import numpy as np

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    print("hello")
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--config_file",
        dest="config_file",
        help="Path to the config files",
        default="configs/MVITv2_L_40x3_train.yaml",
        nargs="+",
    )
    args = parser.parse_args()
    # Setup cfg.
    cfg = get_cfg()
    # if args.cfg_files is not None:
    cfg.merge_from_file(args.config_file)
    print(cfg)
    # model = build_model(cfg)

    print("seeding!!")
    seed_everything(42) #answer to life, universe and everything. hitchhikeers guide to galaxy
    print("seeded!!")
    model = build_rajats_model(cfg)
    print("built model!!")

    optimizer = build_optimizer(model,cfg)
    print("built optimizer!!")

    logger = build_logger(cfg)
    print("built logger !!")
    
    losses = build_losses(cfg)
    print("built losses!!")
    # exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = model.to(device)
    
    
    train_loader, val_loader,test_loader,= build_dataloader(cfg)
    
    max_train_loss = float('inf')
    max_val_loss  = float('inf')
    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = train_single_epoch(cfg,epoch,model,optimizer,losses,logger,device)

        val_loss = test_single_epoch(cfg,epoch,model,optimizer,losses,logger,device)
        
        #save checkpoint nevertheless
        logger.save_checkpoint(model, epoch, len(train_loader), optimizer = None,phase='train')
        print("train loss is ", train_loss, val_loss)
        #save best checkpoint
        if train_loss <  max_train_loss:
            print("yay got train loss down....")
            max_train_loss = train_loss 
            logger.save_best_checkpoint(model, epoch, len(train_loader), optimizer = None, phase='best_train')
        print("returning val loss")
        if val_loss <  max_val_loss:
            max_val_loss = val_loss 
            print("yay got val loss down....")
            logger.save_best_checkpoint(model, epoch, len(val_loader), optimizer = None, phase='best_val')
    # out = model(x)
    # target = torch.zeros_like(out)
    # criterion = nn.BCEWithLogitsLoss()
    # loss = criterion(out.float(),target.float())
    # loss.backward()
    # optimizer.step()
    
    
    #check logger 
    # print("saving...")
    # logger.save_checkpoint(model,10,100 )
    # print("saved!!")
main()