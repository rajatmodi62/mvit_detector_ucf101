#author: rmodi
#this is model interface 
#sends data to correct device 
#perform single forward pass over the model 
# calculate the stats on the meters. 
#print those meters 


#assumes: model arlready sent to device
#data: input of dataloader
#losses: a dict to compute losses 
#returns : total_loss, whichh can be backpropagated, curr_loss which is dict of individual losses. 
import torch
def model_interface(model, data,losses,device):
    
    input_video = data['video'].to(device)
    target_logits = data['label'] 
    target_logits = target_logits.to(device)
    target_seg_mask = data['mask'].to(device)
    # print("mode input", target_logits.shape)
    # exit(1)
    #fwd pass 
    pred_logits, pred_seg_mask  = model(input_video)
    #open losses 

    # print("input", pred_logits.shape,target_logits.shape)
    # exit(1)
    cls_loss = losses['cls_loss'](pred_logits,target_logits)
    # print("1")
    dice_loss = losses['dice_loss'](pred_seg_mask,target_seg_mask)
    # print("2")
    bce_mask_loss = losses['bce_mask_loss'](pred_seg_mask,target_seg_mask)
    # print("3")
    curr_loss = {
        'cls_loss':cls_loss,\
        'dice_loss':dice_loss,\
        'bce_mask_loss':bce_mask_loss,\
    }
    total_loss = 0 
    for k,v in curr_loss.items():
        total_loss+=curr_loss[k]
    
    return pred_logits, pred_seg_mask,total_loss,curr_loss