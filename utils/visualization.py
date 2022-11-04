from pathlib import Path 
import json 
import shutil as sh 
import glob
import pickle 
import numpy as np 
import os 
import  cv2
import copy
import torch

#renormalize a tensor to 0-1

def renormalize_array(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x =  (x- min_val)/(max_val-min_val)
    return x
#frames: list(frames), [(h,w,3)]
def vis_video(frames, masks, save_path):
    frames = [renormalize_array(np.array(frame))*255 for frame in frames]
    masks = [np.array(mask) for mask in masks]
    assert len(frames) > 0    
    frames = copy.deepcopy(frames)
    h, w,_= frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(str(save_path), fourcc, 30, (w,h))
    alpha = 0.5
    frames = [alpha*frames[id] + (1-alpha)*masks[id]*255 for id in range(len(frames))]
    for frame in frames:
        frame = np.uint8(frame)
        video.write(frame)
    video.release()
    print("done")