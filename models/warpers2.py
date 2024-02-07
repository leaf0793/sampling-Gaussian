import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from models.submodule import disparity_regression
from models.warpers import Loss_warper as base_warper
import math
import numpy as np

def normal_distribution_torch(x, mean, sigma):
    return torch.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)


class Loss_warper(base_warper):
    def __init__(self, model=None, maxdisp=192, sigma=1.0):
        super().__init__()
        self.maxdisp = maxdisp
        self.model = model
        self.T_model = None
        self.eval_losses = [
            self.eval_epe,
            self.D1_metric,
        ]
        self.strictness_discreate = True
        # self.sigma=sigma
        self.sigma=1.3
        # sigma = 1.0
        # self.gau_base = math.sqrt(2*np.pi)* sigma
        self.gau_x = torch.Tensor(np.arange(0, 192)).unsqueeze(1).cuda()

    def forward(self, L, R, gt):
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        output = self.model(L, R)
        output = self.unpad_img(output, bot_pad, right_pad)
        mask = (gt > 0) & (gt < self.maxdisp)

        if self.training:
            loss_disp = self.loss_disp(output, gt, mask)
        else:
            loss_disp = [_(output[0], gt, mask) for _ in self.eval_losses]

        return loss_disp, 0

    def groudtruth_to_gaussion(self, mean, sigma=1.0):
        l = mean.shape[0]
        x = self.gau_x.repeat(1, l) # b, 192, l
        ans = torch.exp(-1*((x-mean)**2)/(2*(sigma**2)))/math.sqrt(2*np.pi)* sigma # ans (192, l*b)
        if self.strictness_discreate:
            ans /= torch.sum(ans,dim=0)
        return ans
        # return torch.exp(-1*((x-mean)**2)/(2*(sigma**2)))/math.sqrt(2*np.pi)* sigma


    def loss_disp(self, preds, gt, mask):
        mask = torch.squeeze(mask, dim=1)
        gt = torch.squeeze(gt, dim=1)
        gt_g = self.groudtruth_to_gaussion(gt[mask], sigma=self.sigma)

        if isinstance(preds, list) or isinstance(preds, tuple):
            weights = [0.5, 0.7, 1.0]
            loss1 = []
            eps = 1e-10
            for weights, output in zip(weights, preds):
                output = torch.squeeze(output, dim=1) # B, 192, H, W
                output_ = output.permute(1,0,2,3)[...,mask]
                loss1.append(
                    weights*(
                        # F.kl_div(
                        #         (output_+eps).log(), gt_g+eps,
                        #         reduction='batchmean')
                        F.smooth_l1_loss(
                            output_, gt_g,
                            reduction='sum')
                    )
                )
            loss = sum(loss1)
            # loss = sum(loss1)/mask.shape[0]/torch.cuda.device_count()
            # loss = sum(loss1)/mask.shape[0]/torch.cuda.device_count()
            # loss = sum(loss1)/mask.shape[0]/torch.cuda.device_count()
        else:
            loss = F.smooth_l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss
    



        