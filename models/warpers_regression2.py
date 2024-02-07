import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from models.submodule import disparity_regression
from models.warpers_regression import Loss_warper as Loss_warper_base

# def normal_distribution():
    

class Loss_warper(Loss_warper_base):
    def __init__(self, model=None, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.model = model
        self.eval_losses = [
            self.eval_epe,
            self.D1_metric,
        ]
        self.indice = 5
        

    

    
    def loss_disp(self, preds, gt, mask):
        mask = torch.squeeze(mask, dim=1)
        gt = torch.squeeze(gt, dim=1)
        loss = self.loss_cross(preds[0], gt, mask, diff=preds[1])
        return loss

            
# 