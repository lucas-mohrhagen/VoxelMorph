import torch
from torch import nn
import torch.nn.functional as F


class ImplicitLoss(nn.Module):
    def __init__(self, bce_loss=True):
        super(ImplicitLoss, self).__init__()

        self.bce_loss = bce_loss

    # OCCUPANCY CONSTRAINTS
    def calc_occ_bce_constraint(self, pred_occupancy, gt_occupancy):
        return F.binary_cross_entropy_with_logits(pred_occupancy, gt_occupancy.float(), reduction='mean')


    def forward(self, model_output, gt, epoch, mode='train'):
        loss_constraints = {}

        gt_occupancy = gt['occupancy']
        pred_occupancy = model_output['model_out'].to(torch.float32)

        # OCCUPANCY CONSTRAINTS
        if self.bce_loss:
            loss_constraints['occ_bce_constraint'] = self.calc_occ_bce_constraint(pred_occupancy, gt_occupancy)

        return loss_constraints