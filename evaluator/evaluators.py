#writing the evaluator code 
from einops import rearrange, reduce, repeat
import torch
import numpy as np 


#EVALUATOR FOR A PARTICULAR CLASS, AND IOU INDEX

class Evaluator(object):

  def __init__(self,iou_thresh=0.5):
    
    self.iou_thresh = iou_thresh 
  
    # FOR FRAME IOU 
    self.n_frames = 0 #total no of frames encountered
    self.n_correct_frames = 0 #frames where iou > iou_thresh

    
    #FOR VIDEO IOU
    self.n_vids = 0
    self.vid_level_iou = []

    #FOR CLASS ACCURACY
    self.n_correct = 0 #labels
    self.n_total = 0
  
  def get_fmap(self):
    fmap = (self.n_correct_frames/(self.n_frames +1e-7))*100
    return fmap 
  
  def get_vmap(self):
    #get iou  of indv  tubelets 
    
    n_correct_vids = 0
    for iou in self.vid_level_iou:
      if iou >= self.iou_thresh:
        n_correct_vids +=1
      
    
    return (n_correct_vids/(self.n_vids+1e-7))*100
  
  def get_acc(self):
    return (self.n_correct/(self.n_total+1e-7))*100
  
  
  def get_metrics(self):
    fmap = self.get_fmap()
    vmap = self.get_vmap()
    acc = self.get_acc()
    return acc, fmap, vmap
  #pred_logit : (int) 
  #pred_mask: (clip_len, h, w)
  #assume that this is called on the correct gt class evaluator 

  def run_evaluator(self, pred_logit, gt_logit, pred_mask, gt_mask):

      pred_mask = (pred_mask >0)*1
      gt_mask = (gt_mask>0)*1
      
      if np.sum(gt_mask) ==0:
         return 

      if pred_logit==gt_logit:
        self.n_correct+=1
      
      self.n_total+=1

      
      
      self.n_vids+=1

      t,h,w = pred_mask.shape
      self.n_frames+=t

      intersection = ((pred_mask + gt_mask)==2)*1
      union = ((pred_mask + gt_mask)>0)*1 + 1e-7
      intersection = reduce(intersection, 't h  w -> t','sum')
      union = reduce(union,'t h w -> t', 'sum')  

      iou = intersection/union #[t frames ious]
      #frames satisfying iou creiteria 
      n_correct_iou_frames = np.sum((iou>=self.iou_thresh)*1)

      self.n_correct_frames+=n_correct_iou_frames


      #vid level iou 
      self.vid_level_iou.append(np.sum(intersection)/np.sum(union))


class EvaluatorWrapper(object):
  def __init__(self,N_CLASSES=24):
    self.n_classes = N_CLASSES
    self.evaluators = [Evaluator() for _ in range(N_CLASSES+1)]
  
  def run_wrapper(self, pred_label, gt_label, pred_mask, gt_mask):
    # pred_logit = pred_logit.cpu().numpy()
    # gt_logit = gt_logit.cpu().numpy()
    # pred_mask = pred_mask.cpu().numpy()
    # gt_mask = gt_mask.cpu().numpy()


    b,t,h,w = pred_mask.shape
    for i in range(b):
      b_pred_label = pred_label[i]
      b_pred_mask = pred_mask[i]
      b_gt_label = gt_label[i]
      b_gt_mask = gt_mask[i]
      #print("debug!!", b_pred_label, b_gt_label, b_pred_mask.shape, b_gt_mask.shape)
      #call the evaluator for this particular label 
      self.evaluators[b_gt_label].run_evaluator(b_pred_label, b_gt_label, b_pred_mask, b_gt_mask)
  
  def get_stats(self):
    metrics = {}
    overall_fmap = 0
    overall_vmap = 0 
    overall_acc = 0 

    for cls_id, e in enumerate(self.evaluators):
      if cls_id==0:
        continue
      acc,fmap,vmap = e.get_metrics()

      metrics[cls_id] = {}
      metrics[cls_id]['fmap'] = fmap
      metrics[cls_id]['vmap'] = vmap
      metrics[cls_id]['acc'] = acc
      
      overall_fmap+=fmap
      overall_vmap+=vmap
      overall_acc+=acc

    #print("overall", overall_fmap, overall_vmap, overall_acc,len(self.evaluators))
    # DONT COUNT BG CLASS
    metrics["fmap"] = overall_fmap/(len(self.evaluators)-1)
    metrics["vmap"] = overall_vmap/(len(self.evaluators)-1)
    metrics["acc"] = overall_acc/(len(self.evaluators)-1)

    return metrics


def build_evaluator(cfg=None):

  evaluator = EvaluatorWrapper()
  return evaluator 

if __name__ == "__main__":

    evaluator = build_evaluator()
    BS = 4
    T = 16
    H = 224
    W = 224
    N_CLASSES = 24
    pred_label = torch.randint(0,N_CLASSES+1,(BS,))
    gt_label = pred_label#torch.randint(0,N_CLASSES+1,(BS,))
    
    pred_mask = (torch.randn(BS,T,H,W) >0.5)*1
    gt_mask = (torch.randn(BS, T,H, W) >0.5)*1


    evaluator.run_wrapper(pred_label, gt_label, pred_mask, gt_mask)

    evaluator.get_stats()