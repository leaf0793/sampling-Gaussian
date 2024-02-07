from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
from tqdm import tqdm
from utils.kittiColor import kitti_colormap
from utils.train_util import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
# parser.add_argument('--loadmodel', default='zoo_zoo/backbone/slim_1.0/0723_30_0.6263.tar',
parser.add_argument('--loadmodel', default='zoo_zoo2/backbone/first_second_kitti/1018_0900_0.6277.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= '/data/datasets/Sceneflow/frames_finalpass/TEST/A/0000/left/0006.png',
                    help='load model')
parser.add_argument('--rightimg', default= '/data/datasets/Sceneflow/frames_finalpass/TEST/A/0000/right/0006.png',
                    help='load model')                                      
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--maxval', type=int, default=192)
parser.add_argument('--Sceneflow', type=int, default=1)
parser.add_argument('--savegtori', type=int, default=1)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
Sceneflow=bool(args.Sceneflow)
if Sceneflow:
    # args.loadmodel = 'zoo_zoo2/try2/from_amax_sigma_2gpu_1.3/1124_0020_0.6257.tar'
    # args.loadmodel = 'zoo_zoo2/try2/from_amax_kldiv/1105_0020_0.8093.tar'
    args.loadmodel = 'zoo_zoo2/try2/from_amax_1.0_kitti/1121_0006_0.9283.tar'
    # args.loadmodel = 'zoo_zoo2/try2/from_amax_sigma_2gpu_1.0/1121_0013_0.8060.tar'
else:
    args.loadmodel = 'zoo_zoo2/try2/from_amax_sigma_2gpu_1.3_kitti/1227_1000_0.3248.tar'
    # args.loadmodel = 'zoo_zoo2/try2/from_amax_1.0_kitti/1107_1000_0.7311.tar'
# args.loadmodel = 'zoo_zoo2/try2/from_amax_sigma_2gpu_1.3_kitti/1227_1000_0.3248.tar'

args.leftimg = "/data/datasets/KITTI/data_scene_flow/testing/image_2/000024_10.png"
args.rightimg = "/data/datasets/KITTI/data_scene_flow/testing/image_3/000024_10.png"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# from models.official_model.stackhourglass import PSMNet
from models.official_model_try2.stackhourglass import PSMNet
# from models.official_model_try1.stackhourglass import PSMNet
# if args.model == 'stackhourglass':
model = PSMNet(args.maxdisp)

model.eval()
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

# if args.loadmodel is not None:
print('load PSMNet')
state_dict = torch.load(args.loadmodel)
updated = state_dict['state_dict']
updated = {k.replace('.model',''): v for k,
                    v in updated.items()}

model.load_state_dict(updated)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    with torch.no_grad():
        disp = model(imgL,imgR)[0]

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

def test2(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp = model(imgL,imgR)
            disp = traslator(disp)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp



def traslator(output):
    diff = output[1]
    int_part = torch.argmax(F.log_softmax(output[0], dim=1), dim=1)
    m_ = F.one_hot(int_part, num_classes=48).permute(0, 3, 1, 2) # B, 48, H,W
    m_ = m_.type('torch.cuda.BoolTensor')
    
    diff_p1 = diff[:, :5, :, :, :] # B, 5, D,H,W
    diff_p1 = torch.argmax(
        F.log_softmax(diff_p1, dim=1),
        dim=1)  # B,D,H,W
    diff_p1[~m_] = 0
    diff_p1 = torch.sum(diff_p1, dim=1) # B, H,W

    diff_p2 = diff[:, 5, :, :, :]  # B, 1, D, H, W 
    diff_p2 = F.tanh(diff_p2).squeeze(1) # B, D, H, W 
    diff_p2[~m_] = 0
    diff_p2 = torch.sum(diff_p2, dim=1) # B, H, W
    
    # real_output = int_part*4 + diff_p1 + diff_p2
    real_output = int_part*4+ diff_p1

    return real_output



def use_kitti():
    print(" use kitti")
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])
    l = []
    for R, T, S in os.walk('/data/datasets/KITTI/data_scene_flow/training/image_2'):
        for f in S:
            if '.jpg' or '.png' in f:
                if '_10' in f:
                    l.append([
                        R+'/'+f,
                        R.replace('image_2', 'image_3')+'/'+f,
                    ])
            leftimg_rightimg=[]
    
    for leftimg, rightimg in tqdm(sorted(l)):
        imgL_o = Image.open(leftimg).convert('RGB')
        imgR_o = Image.open(rightimg).convert('RGB')
        dir = "submission/"+f'/{leftimg.split("/")[-1][:-4]}_ori.png'
        disp_l = cv2.imread(leftimg.replace('image_2','disp_occ_0'),-1)/256
        disp_l = kitti_colormap(disp_l, maxval=args.maxval)
        # img = kitti_colormap(img, maxval=args.maxval)
        # dir.replace('image_2','disp_occ_0')
        dir2 = dir.replace('submission','submission/colgt_SF')
        imgL_o.save(dir2)
        cv2.imwrite(dir2.replace('_ori','_gt'),disp_l)
        # cv2.imwrite("submission/colgt_SF/"+f'/{batch_idx:04}_gt.png', disp_L)
        # imgL_o.save(dir)
        
        
        # cv2.imwrite("submission/"+f'/{batch_idx:04}_gt.png', imgL_o)

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        imgL = imgL.cuda()
        imgR = imgR.cuda()     

        leftimg_rightimg.append(
            [imgL, imgR, leftimg, top_pad, right_pad ]
        )

    return leftimg_rightimg

def use_Sceneflow(lenth = 200):
    print(" use Sceneflow")
    from datasets.dataset import SceneFlowDataset as DATASET
    list_filename_test = "filenames/sceneflow_test_fly.txt"
    # list_filename_test = "filenames/sceneflow_train_fly.txt"

    Test_Dataset = DATASET(
        training=False, want_size=(0,0), list_filename=list_filename_test, mode='val',
        server_name="LARGE",
    )
    TestImgLoader = torch.utils.data.DataLoader(
        Test_Dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        drop_last=False,
        collate_fn=BatchCollator(),
        sampler=None,
    )
    def f(a): return a.cuda(0) if a is not None else a

    l = []

    for batch_idx, data_batch in enumerate(TestImgLoader):
        imgL = f(data_batch['left'])
        imgR = f(data_batch['right'])
        right_pad = 4
        imgL = F.pad(imgL,(0,0, 0,right_pad))
        imgR = F.pad(imgR,(0,0, 0,right_pad))
        l.append([imgL, imgR, f'/{batch_idx:04}.png', 0, right_pad])
        
        # dir2 = dir.replace('submission','/colgt_SF')
        # imgL_o.save(dir2)
        # cv2.imwrite(dir2.replace('_ori','_gt'),disp_l)
        if args.savegtori:
            disp_L = f(data_batch['disp']).squeeze(0)
            disp_L = torch.squeeze(disp_L, dim=0).data.cpu().numpy()
            disp_L = kitti_colormap(disp_L, maxval=args.maxval)
            cv2.imwrite("submission/colgt_SF/"+f'/{batch_idx:04}_gt.png', disp_L)

        if batch_idx>lenth:
            break
        
        
    return l



def re_crop(pred_disp,top_pad,right_pad):

    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        img = pred_disp[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp
    return img

def torch_2_img(imgL,top_pad,right_pad):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    imgL = torch.squeeze(imgL).permute(1,2,0).data.cpu().numpy()
    imgL*= np.array(normal_mean_var['std'])
    imgL+=  np.array(normal_mean_var['mean'])
    imgL*=255
    # data = imgL.data.cpu().numpy()
    if top_pad !=0 and right_pad != 0:
        img = imgL[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        img = imgL[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        img = imgL[top_pad:,:]
    else:
        img = imgL
    return img
    


def main():
        if Sceneflow:
            leftimg_rightimg = use_Sceneflow()
        else:
            leftimg_rightimg = use_kitti()

        # TrainImgLoader_disp, TestImgLoader_disp = DATASET_disp(cfg)

        
        for imgL,imgR,leftimg,top_pad, right_pad in tqdm(leftimg_rightimg):

            pred_disp = test(imgL,imgR)
            img = re_crop(pred_disp,top_pad, right_pad)
            imgL = torch_2_img(imgL,top_pad, right_pad)
            # dir.replace('submission','/colgt_SF')
            dir = "submission/"+leftimg.split('/')[-1]

            if Sceneflow and args.savegtori:
                cv2.imwrite(dir.replace('submission','submission/colgt_SF').replace('.png','_ori.png'), imgL)
            if 0:
                img = kitti_colormap(img, maxval=args.maxval)
                cv2.imwrite(dir, img)
                continue
            img = (img*256).astype('uint16')
            img = Image.fromarray(img)
            img.save(dir)

if __name__ == '__main__':
   main()
