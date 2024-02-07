import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from models.submodule import disparity_regression


class Loss_warper(nn.Module):
    def __init__(self, model=None, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.model = model
        self.eval_losses = [
            self.eval_epe,
            self.D1_metric,
        ]
        self.indice = 5
        
    def activ_func(self, x):        
        # diff_p2 = F.tanh(diff_p2).squeeze(1) # B, D, H, W 
        # diff_p2 = F.sigmoid(diff_p2).squeeze(1)  # B,D,H,W
        # diff_p2 = diff_p2*2-1
        x = F.tanh(x)
        # x = F.tanh(x)*0.5
        # x = F.sigmoid(x)
        # x = F.sigmoid(x)*2-1
        # x = torch.clamp(x, -1, 1)
        return x
    
    def forward(self, L, R, gt):
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        output = self.model(L, R)
        output = self.unpad_img(output, bot_pad, right_pad)
        mask = (gt > 0) & (gt < self.maxdisp)

        if self.training:
            loss_disp = self.loss_disp(output, gt, mask)
        else:
            real_output = self.traslator(output)
            loss_disp = [_(real_output, gt, mask) for _ in self.eval_losses]

        return loss_disp, 0
    
    
    def submission(self, L, R):
        # L, R, bot_pad, right_pad = self.pad_img(L, R)
        output = self.model(L, R)
        # output = self.unpad_img(output, bot_pad, right_pad)
        real_output = self.traslator(output)

        return real_output
    
    def traslator(self, output):
        diff = output[1]
        int_part = torch.argmax(F.log_softmax(output[0], dim=1), dim=1)
        m_ = F.one_hot(int_part, num_classes=48).permute(0, 3, 1, 2) # B, 48, H,W
        m_ = m_.type('torch.cuda.BoolTensor')
        
        diff_p1 = diff[:, :self.indice, :, :, :] # B, 5, D,H,W
        diff_p1 = torch.argmax(
            F.log_softmax(diff_p1, dim=1),
            dim=1)  # B,D,H,W
        diff_p1[~m_] = 0
        diff_p1 = torch.sum(diff_p1, dim=1) # B, H,W

        diff_p2 = diff[:, self.indice, :, :, :]  # B, 1, D, H, W 
        # diff_p2 = F.tanh(diff_p2).squeeze(1) # B, D, H, W 
        diff_p2 = self.activ_func(diff_p2).squeeze(1)
        diff_p2[~m_] = 0
        diff_p2 = torch.sum(diff_p2, dim=1) # B, H, W
        
        # real_output = int_part*4 + diff_p1
        real_output = int_part*4 + diff_p1 + diff_p2
        # real_output = int_part*4

        # real_output = torch.argmax(F.log_softmax(output[0]), dim=1)+\
        #         torch.argmax(F.log_softmax(output[1][:, :4, :, :, :]), dim=1)
        return real_output
    

    def pad_img(self, L, R, base=32):
        if self.training is True:
            return L, R, 0, 0
        else:
            bot_pad = int(
                base-L.shape[2] % base) if int(base-L.shape[2] % base) != base else 0
            right_pad = int(
                base-L.shape[3] % base) if int(base-L.shape[3] % base) != base else 0
            # self.model.Regression.set_full_shape(
            #     (L.shape[2]+bot_pad, L.shape[3]+right_pad))
            L = F.pad(
                L, (0, right_pad, 0, bot_pad),
                "constant", 0
            )
            R = F.pad(
                R, (0, right_pad, 0, bot_pad,),
                "constant", 0
            )
            return L, R, bot_pad, right_pad

    def unpad_img(self, output, bot_pad, right_pad):
        if self.training is True:
            return output
        else:
            # output = list(output)
            for i in range(len(output)):
                if output[i] is not None:
                    output[i] = output[i][..., :-bot_pad,:] if bot_pad > 0 else output[i]
                    output[i] = output[i][..., :-
                                          right_pad] if right_pad > 0 else output[i]
            return output

    def eval_epe(self, preds, gt, mask):

        loss = F.l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss

    
    def D1_metric(self, D_es, D_gt, mask):
        tmp(D_es, D_gt, mask)
        D_es, D_gt = D_es[mask], D_gt[mask]
        E = torch.abs(D_gt - D_es)
        # tmp = torch.mean(E)
        err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
        return torch.mean(err_mask.float())


    def loss_disp(self, preds, gt, mask):
        mask = torch.squeeze(mask, dim=1)
        gt = torch.squeeze(gt, dim=1)

        if isinstance(preds, list) or isinstance(preds, tuple):
            # weights = [0.5, 0.7, 1.0]
            loss1 = [
                0.5*self.loss_cross(preds[0], gt, mask),
                0.7*self.loss_cross(preds[1], gt, mask),
                1.0*self.loss_cross(preds[2], gt, mask, diff=preds[3])
            ]

            loss = sum(loss1)
        else:
            loss = self.loss_cross(preds[0], gt, mask, diff=preds[1])
        return loss
    
    def loss_cross(self, out, gt, mask, diff=None):
        out = F.log_softmax(out, dim=1)
        gt_ = torch.trunc(gt/4)
        gt_ = gt_.type('torch.cuda.LongTensor')
        gt_[~mask] = -1
        loss = F.nll_loss(out, gt_, ignore_index=-1)

        if diff is None:
            return loss
        else:
            gt_2 = torch.floor(gt/4)
            depth_g = gt - gt_2*4
            # print(torch.max(depth_g,1)[0])
            # print(torch.max(depth_g,1)[1])
            # print(torch.where(depth_g == 4.0, depth_g, 0))
            # print(((depth_g == 4).nonzero(as_tuple=True)[0]))
            # out = F.log_softmax(out, dim=1)
            # gt = torch.trunc(gt/4)
            
            mask2 = (gt_2 - torch.argmax(out, dim=1)) < 2
            mask2 = mask2 & mask  # B,H,W

            if len(mask2 == 1) > 0:
                gt_copy = gt_2.type('torch.cuda.LongTensor')
                gt_copy[~mask2] = -1
                gt_copy += 1
                gt_copy = F.one_hot(gt_copy, num_classes=49).permute(0,3,1,2)[:,1:,:,:]
                gt_copy = gt_copy.type('torch.cuda.BoolTensor')

                d_g1 = torch.floor(depth_g)
                d_g2 = depth_g - d_g1
                d_g1 = d_g1.type('torch.cuda.LongTensor')
                d_g1 = d_g1.unsqueeze(1).repeat(1, 48, 1, 1)
                d_g1[~gt_copy]=-1
                # print(torch.max(d_g1))
                
                diff_p1 = diff[:, :self.indice, :, :, :]
                diff_p1 = F.log_softmax(diff_p1, dim=1) # B,C,D,H,W
                
                diff_p2 = diff[:, self.indice, :, :, :]
                # diff_p2 = F.tanh(diff_p2).squeeze(1) # B,D,H,W
                diff_p2 = self.activ_func(diff_p2).squeeze(1)
                # diff_p2 = F.sigmoid(diff_p2).squeeze(1)  # B,D,H,W
                # diff_p2 = diff_p2*2-1

                loss2 = F.nll_loss(
                    diff_p1, d_g1, ignore_index=-1)
                loss3 = F.smooth_l1_loss(
                    diff_p2[gt_copy], d_g2[mask2])
               
                loss = loss+loss2+loss3
                
            return loss
            
# 
# global idx
# idx=0
def tmp(D_es, D_gt, mask):
    err = D_es- D_gt
    err[~mask]=0
    import cv2
    import numpy as np
    import random
    err = err.cpu().squeeze().numpy()
    mask1 = (err<1) &(err>0)
    mask2 = (err>1) &(err<2)
    mask3 = (err>2) &(err<3)
    mask4 = (err>3)
    new_mat = np.zeros((384, 1248, 3), dtype=np.uint8)
    # print(new_mat.shape)
    new_mat[mask1] = (255,0,0)
    new_mat[mask2] = (0,255,0)
    new_mat[mask3] = (0,0,255)
    new_mat[mask4] = (255,255,255)
    # err = (err*256*200).astype('uint16')
    cv2.imwrite(f"errpng/{random.randint(0,1000)}.png", new_mat)
    # idx+=1