#author: rmodi, 
#implements the logger class
# saves the logs to the correct place
# dumps the model checkpoints to that location which is created by wandb 
# but, prevents upload of saved checkpoints to the wandb server 


import wandb 
import torch 
import numpy as np 
import os
from pathlib import Path
import json 
class Logger(object):
    def __init__(self,cfg):
        #start wandb 
        wandb.init(project = cfg.PROJECT_NAME,config = cfg)
        self.run_dir = Path(wandb.run.dir)


    #save checkpoint at a particular epoch 
    def save_checkpoint(self, model, epoch, iteration, optimizer = None,phase='train'):
        assert model is not None
        name = "checkpoint_epoch_{}_iteration_{}.pth".format(epoch, iteration)
        save_dir  = self.run_dir/phase 
        print("save_dir",save_dir)
        if not os.path.isdir(str(save_dir)):
            save_dir.mkdir(exist_ok=True, parents=True)
        save_dir  = self.run_dir/phase/name 
        if optimizer:

            save_dict = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),                    
                        }
        else:
            save_dict = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':None,                   
                        }
        torch.save(save_dict,str(save_dir) )

    #load the checkpoint back 
    def load_checkpoint(self,model, optimizer, model_path=None,strict = True):
        assert model_path is not None 
        
        checkpoint = torch.load(str(model_path))
        state_dict = checkpoint["model_state_dict"]
        #load strictly by default 
        model.load_state_dict(state_dict, strict = strict) 
        return 

    #save best checkpoint 
    def save_best_checkpoint(self, model, epoch, iteration, optimizer = None, phase='best_train'):
        name = "checkpoint_epoch_{}_iteration_{}.pth".format(epoch, iteration)
        save_dir = self.run_dir/phase
        if not os.path.isdir(str(save_dir)):
            save_dir.mkdir(exist_ok=True, parents=True)
        save_dir = self.run_dir/phase/name
        if optimizer:

            save_dict = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),                    
                        }
        else:
            save_dict = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':None,                   
                        }
        torch.save(save_dict,str(save_dir))

    def dump_dict(self, content, name = 'results.json'):
        save_dir = self.run_dir/name
        out_file = open(str(save_dir),'w')
        json.dump(content,out_file)
        print("dumped dict!!")

#to do : hahve cfg dependence layer 
def build_logger(cfg=None):
        if cfg:
            return Logger(cfg)
        else:
            return Logger()