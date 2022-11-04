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
import random, os
import numpy as np
from models.model_interface import model_interface
import torch.nn.functional as F
from utils.visualization import vis_video
from einops import rearrange, reduce, repeat
from evaluator.evaluators import build_evaluator

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
        default="configs/MVITv2_L_40x3_test.yaml",
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
    

    logger = build_logger(cfg)
    print("built logger !!")

    losses = build_losses(cfg)
    print("built losses!!")
    
    model = build_rajats_model(cfg)
    print("built model!!")

    print("loading weights..")
    logger.load_checkpoint(model, optimizer=None, model_path=cfg.CHECKPOINT_PATH,strict = True)
    print("loaded weights!!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader, val_loader,test_loader,= build_dataloader(cfg)
    print("built dataloader!!")
    
    print("building evaluator..")
    evaluator = build_evaluator()
    print("built evaluator!!")
    

    print("building meters..")
    meters = {
    'dice_meter':AverageMeter(),\
    'bce_meter':AverageMeter(),\
    'cls_meter':AverageMeter(),\
    }
    print("built meters!!")

    model.eval()
    with torch.no_grad():
        for index, data in enumerate(test_loader):
            print("Doing {}/{}".format(index, len(test_loader)))
            pred_logits, pred_seg_mask,total_loss,curr_loss_dict = model_interface(model, data, losses, device)
            
            
            bs = pred_logits.shape[0] 
            pred_logits = torch.argmax(pred_logits, 1)
            
            pred_seg_mask = F.sigmoid(pred_seg_mask)
            pred_seg_mask = (pred_seg_mask>=0.5)*1

            #convert everything to numpy 
            pred_logits = pred_logits.detach().cpu().numpy()
            target_logits = data['label'].detach().cpu().numpy()
            pred_seg_mask = pred_seg_mask.detach().cpu().numpy()
            gt_seg_mask = data['mask'].detach().cpu().numpy()
            gt_frames = data['video']
            

            cls_loss = curr_loss_dict['cls_loss']
            dice_loss = curr_loss_dict['dice_loss']
            bce_mask_loss = curr_loss_dict['bce_mask_loss']
            meters['cls_meter'].update(cls_loss,bs)
            meters['dice_meter'].update(dice_loss,bs)
            meters['bce_meter'].update(bce_mask_loss,bs)

            #prepare stuff for vis################################## 
            # b,t,h,w = pred_seg_mask.shape    
            # frames = gt_frames[0]
            # masks = pred_seg_mask[0]
        
            # print(curr_loss_dict)
            # print("shapes of chosen frames and mask", frames.shape, masks.shape)
            # print("logits", pred_logits, target_logits)
            # vis_frames = [frames[:,i,:,:] for i in range(t)]
            # vis_masks = [masks[i,:,:] for i in range(t)] 

            # for i in range(t):
            #     vis_frames[i] = rearrange(vis_frames[i], 'c h w -> h w c')
            #     vis_masks[i] = repeat(vis_masks[i], 'h w -> c h w', c=3)
            #     vis_masks[i] = rearrange(vis_masks[i], 'c h w -> h w c')
            
            # vis_video(vis_frames, vis_masks,'./krishna.mp4')
            # print("saved video!!")
            ###########################################################
            evaluator.run_wrapper(pred_logits, target_logits, pred_seg_mask, gt_seg_mask)
            
            if index%cfg.PRINT_FREQ ==0:
                print("Done Epoch {}/{} {}/{} Class Loss:{:.2f}, Dice Loss:{:.3f} Bce Loss:{:.3f}".format(
                    1, 
                    cfg.NUM_EPOCHS,
                    index+1,
                    len(test_loader),
                    meters['cls_meter'].avg, 
                    meters['dice_meter'].avg, 
                    meters['bce_meter'].avg
                ))
            if index%1000 ==0:
                metrics = evaluator.get_stats()
                print(metrics)
            wandb.log({"progress/test/epoch": 1})
            wandb.log({"progress/test/idx": index})
            wandb.log({"test/cls_loss": meters['cls_meter'].avg})
            wandb.log({"test/dice_loss": meters['dice_meter'].avg})
            wandb.log({"test/bce_mask_loss": meters['bce_meter'].avg})
            
            #break
        metrics = evaluator.get_stats()
        logger.dump_dict(metrics)

        print(metrics)
        print("testing complete!!")
            


main()