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
from tqdm import tqdm
# from models import *
from PIL import Image
from models.official_model import *
from utils.kittiColor import kitti_colormap
from utils.train_util import *
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/data2/dataset/kitti/data_scene_flow/testing/',
                    help='select model')
# parser.add_argument('--loadmodel', default='official_zoo/pretrained_model_KITTI2015.tar',
parser.add_argument('--loadmodel', default='zoo_zoo/backbone/slim_1.0_kitti/1018_1000_0.4354.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--maxval', type=int, default=192)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
Sceneflow = True
if Sceneflow:
    args.loadmodel='/home/ubtnavi/pby/PSMnet-NewRegression/zoo_zoo/backbone/slim_1.0/0727_20_0.7705.tar'
else:
    args.loadmodel='/home/ubtnavi/pby/PSMnet-NewRegression/zoo_zoo/backbone/slim_1.0_kitti/1018_1000_0.4354.tar'
    
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# if args.KITTI == '2015':
#    from dataloader import KITTI_submission_loader as DA
# else:
#    from dataloader import KITTI_submission_loader2012 as DA  

# test_left_img, test_right_img = DA.dataloader(args.datapath)
# with open('./filenames/kitti15_test.txt', 'r') as Fs:
#     da = Fs.readlines()
# del Fs
# test_left_img, test_right_img = [], []
# for d in da:
#     d = d.strip().split(' ')
#     test_left_img.append(d[0])
#     test_right_img.append(d[1])
# test_left_img, test_right_img = DA.dataloader(args.datapath.replace('testing','training'))



# model = nn.DataParallel(model, device_ids=[0])
from models.official_model.stackhourglass import PSMNet
model = PSMNet(192)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    if 'official_zoo' in args.loadmodel:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
    else:
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
        output = model(imgL,imgR)[0]
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def use_kitti():
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
            [imgL,imgR, leftimg, top_pad, right_pad ]
        )

    return leftimg_rightimg

def use_Sceneflow(lenth = 200):
    
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
        l.append([imgL, imgR, f'/{batch_idx:04}.png', 0, right_pad ])
        if batch_idx>lenth:
            break
        
        
    return l



def main():
        if Sceneflow:
            leftimg_rightimg = use_Sceneflow()
        else:
            leftimg_rightimg = use_kitti()

        
        for imgL,imgR,leftimg,right_pad,top_pad in tqdm(leftimg_rightimg):
            pred_disp = test(imgL,imgR)

            # print('time = %.2f' %(time.time() - start_time))


            if top_pad !=0 and right_pad != 0:
                img = pred_disp[top_pad:,:-right_pad]
            elif top_pad ==0 and right_pad != 0:
                img = pred_disp[:,:-right_pad]
            elif top_pad !=0 and right_pad == 0:
                img = pred_disp[top_pad:,:]
            else:
                img = pred_disp
            dir = "submission_legend/"+leftimg.split('/')[-1]
            if True:
                img = kitti_colormap(img, maxval=args.maxval)
                # img = cv2.applyColorMap(img, colormap=cv2.COLORMAP_JET)
                cv2.imwrite(dir, img)
                continue
            img = (img*256).astype('uint16')
            img = Image.fromarray(img)
            img.save("submission/"+leftimg.split('/')[-1])

if __name__ == '__main__':
    main()






