#assumes semantic segmentation treatment of the action detection problem 
import os
import cv2
import pickle
import numpy as np
import collections 
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from glob import glob
from pathlib import Path
from einops import rearrange, reduce, repeat
#import transforms 
import pytorchvideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    RandomResizedCrop
) 
from utils.visualization import vis_video
import random 

def read_pickle(path):
    with open(path, 'rb') as fid:
            ann = pickle.load(fid, encoding='iso-8859-1')
    return ann


#make a transform
# To DO: overload with crop size later on 
def make_transform(mode, cfg= None):
    TRAIN_CROP_SIZE = cfg.DATA.TRAIN_CROP_SIZE if cfg is not None else 224
    TEST_CROP_SIZE  = cfg.DATA.TEST_CROP_SIZE if cfg is not None else 224
    MEAN = cfg.DATA.MEAN if cfg is not None else [0.485, 0.456, 0.406]
    STD = cfg.DATA.STD if cfg is not None else [0.485, 0.456, 0.406]

    
    if mode =='train':
        print("train transform..")
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    
                    NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomHorizontalFlipVideo(0.5),
                    RandomResizedCrop(TRAIN_CROP_SIZE, TRAIN_CROP_SIZE,[1,1],[1,1]),
                    
                ]
            ),
        )
    else:
        # for both test /val 
        print("test transform..")  
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    
                    NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    CenterCropVideo(TEST_CROP_SIZE),
                ]
            ),
        )
    return transform
class VideoDataset(Dataset):

    def __init__(self, frame_path, ann_path, semantic_masks, transform=None, clip_len=16, crop_size=224,
                 mode='train'):
        pass

        self.frame_path = Path(frame_path) 
        self.ann_path = Path(ann_path) 
        self.semantic_masks = Path(semantic_masks)
        self.transform = transform 
        self.K = clip_len #tubelet length extracted 
        self.crop_size = crop_size 
        self.mode = mode #train/test 
        #print("loading ann..")
        self.ann = read_pickle(ann_path)
        #print("loaded")
        self.vids = [] 
        if mode=='train':
            self.vids = sorted(self.ann['train_videos'][0])
        elif mode=='test':
            self.vids = sorted(self.ann['test_videos'][0])
        elif mode=='val':
            self.vids = sorted(self.ann['test_videos'][0])#[:700]
            self.mode =  'test'
        self.labels = self.ann['labels']
        self.class_to_label = {}
        self.class_to_label[0] = 'no_class'
        for idx, cls in enumerate(self.labels):
            self.class_to_label[cls] = idx + 1
        

        #contains the video id, along with the keyframe id to sample
        self.dataset = [] 
        for vid in self.vids:
            n_frames = self.ann['nframes'][vid]
            sampled_indices = [i for i in range(1, n_frames-self.K+2)]
            for idx in sampled_indices:
                self.dataset.append((vid,idx)) #keyframe 

        
        #take a smaller size of dataset 
        if mode!='test':
            #need smaller dataset for training
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:2500]
    
    def __getitem__(self, index):
        
        v_id, keyframe = self.dataset[index]
        # print(v_id, keyframe)
        # v_id = 'IceDancing/v_IceDancing_g07_c05'
        # keyframe = 23
        #read frames
        frames = []
        for id in range(keyframe,keyframe+self.K):
            # print(str(self.frame_path/v_id/(str(id).zfill(5)+'.jpg')))
            frames.append(cv2.imread(str(self.frame_path/v_id/(str(id).zfill(5)+'.jpg'))))

        #read masks 
        masks = []
        for id in range(keyframe,keyframe+self.K):
            # print(str(self.frame_path/v_id/(str(id).zfill(5)+'.jpg')))
            masks.append(cv2.imread(str(self.semantic_masks/self.mode/v_id/(str(id).zfill(5)+'.jpg'))))
        #threshold masks 
        masks = [(mask>120)*1 for mask in masks]
        #print("unqiue",np.unique(masks[0]))
        video_volume = np.stack(frames+masks)
        video_volume = rearrange(video_volume,'t h w c-> c t h w' )
        d = {'video': torch.Tensor(video_volume)}
        if self.transform:
            d = self.transform(d)
        #print("after transform",d.keys(),d['video'].shape)
        video_volume = d['video']
        c,t,h,w = video_volume.shape
        #print(t)
        video,masks = video_volume[:,:(t//2),:,:],video_volume[:,(t//2):,:,:]
        
        masks = (masks>0)*1




        ###### UNCOMMENT THIS CODE FOR VISUALIZATION 
        # reshape mask 
        # check mask 
        # mask = masks[:,0,:,:].numpy()
        # mask = rearrange(mask, 'c h w -> h w c')
        # print("after transform", np.unique(mask))
        # cv2.imwrite('./mask.jpg', np.uint8(mask*255))

        # video = rearrange(video, 'c t h w -> t h w c')
        # masks = rearrange(masks, 'c t h w-> t h w c')
        
        # viz_video = [video[i] for i in range(video.shape[0])]
        # viz_mask = [masks[i] for i in range(masks.shape[0])]
        # vis_video(viz_video, viz_mask, './test_krishna.mp4')
        # exit(1)
        ##################################################


        #choose onnly one channel 
        masks = masks[0]
        # print("sum is ", torch.sum(masks))
        if torch.sum(masks)>=1:
            class_id = v_id.split('/')[0]
            label = self.class_to_label[class_id]
        else:
            label = 0
        # print("label is", label)
        # print(masks.shape, video.shape)
        target ={
            'video' : video,\
            'mask': masks,\
            'label':label,\
            'keyframe':keyframe,\
            'video_id':v_id,\
            'dataset_size':len(self.dataset),\
        }
        # target ={
        #     'video' : torch.as_tensor(video),\
        #     'mask': torch.as_tensor(masks),\
        #     'label':label,\
        #     'keyframe':torch.as_tensor(keyframe),\
        #     'video_id':v_id,\
        #     'dataset_size':len(self.dataset),\
        # }
        
        
        
        return target
        pass


    def __len__(self):
        return  len(self.dataset)

def build_dataloader(cfg):
    #dataset
    train_dataset = VideoDataset( mode = 'train',\
                            frame_path = 'datasets/ucf24/rgb-images',\
                            ann_path = 'datasets/ucf24/UCF101v2-GT.pkl',\
                            semantic_masks = 'datasets/semantic_seg_mask_ucf',\
                            transform = make_transform("train",cfg),\
                            clip_len = cfg.DATA.NUM_FRAMES
                         )
    val_dataset = VideoDataset( mode = 'val',\
                            frame_path = 'datasets/ucf24/rgb-images',\
                            ann_path = 'datasets/ucf24/UCF101v2-GT.pkl',\
                            semantic_masks = 'datasets/semantic_seg_mask_ucf',\
                            transform = make_transform("val",cfg),\
                            clip_len = cfg.DATA.NUM_FRAMES
                         )

    test_dataset = VideoDataset( mode = 'test',\
                            frame_path = 'datasets/ucf24/rgb-images',\
                            ann_path = 'datasets/ucf24/UCF101v2-GT.pkl',\
                            semantic_masks = 'datasets/semantic_seg_mask_ucf',\
                            transform = make_transform("test",cfg),\
                            clip_len = cfg.DATA.NUM_FRAMES
                         )
    #loader
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE,\
                                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
                                               pin_memory=False)
                                               
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                        batch_size=cfg.DATA_LOADER.TEST_BATCH_SIZE, 
                                        shuffle=False,
                                        num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
                                        pin_memory=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=cfg.DATA_LOADER.TEST_BATCH_SIZE, 
                                        shuffle=False,
                                        num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
                                        pin_memory=False)

    
    
    return train_loader, val_loader,test_loader

if __name__ == "__main__":
    
    # transform = make_transform('train')
    # dataset = VideoDataset( mode = 'train',\
    #                         frame_path = 'datasets/ucf24/rgb-images',\
    #                         ann_path = 'datasets/ucf24/UCF101v2-GT.pkl',\
    #                         semantic_masks = 'datasets/semantic_seg_mask_ucf',\
    #                         transform = transform,\
    #                         clip_len = 8
    #                      )
    # print(len(dataset))
    # while 1:
    #     dataset[180000]
    from configs.defaults import get_cfg
    import argparse
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
    train_loader, val_loader,test_loader= build_dataloader(cfg)
    print(len(train_loader),len(test_loader))
    # for i, item in enumerate(train_loader):
    #     print("fetch!!",i,"/", len(train_loader))
        #print(type(item),item['video'].shape)
    for i, item in enumerate(test_loader):
        print("fetch!!",i,"/", len(train_loader))
