from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import math

from scipy.stats import norm


def criterion_alternative_l2(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


class Overhual(nn.Module):
    """
    A Comprehensive Overhaul of Feature Distillation, ICCV 2019
    """

    def __init__(self):
        super(Overhual, self).__init__()

    def get_margin_from_BN(self, bn):
        margin = []
        std = bn.weight.data
        mean = bn.bias.data
        for (s, m) in zip(std, mean):
            s = abs(s.item())
            m = m.item()
            if norm.cdf(-m / s) > 0.001:
                margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
            else:
                margin.append(-3 * s)

        return torch.FloatTensor(margin).to(std.device)
        # return margin

    def forward(self, g_s, g_t, teacher_bns, each):
        feat_num = len(g_t)
        batch_size = g_s[0].shape[0]

        if each:
            weights = []
            for i in range(feat_num):  # if have 3 blocks
                if i < feat_num // 3:
                    weights.append(2 ** 2)
                elif i < 2 * feat_num // 3:
                    weights.append(2 ** 1)
                else:
                    weights.append(2 ** 0)
        else:
            weights = [2 ** (feat_num - i - 1) for i in range(feat_num)]

        margins = [self.get_margin_from_BN(bn=bn) for bn in teacher_bns]
        margins = [margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach() for margin in margins]

        losses = [criterion_alternative_l2(s, t, m) for s, t, m in zip(g_s, g_t, margins)]
        losses = [l / w for w, l in zip(weights, losses)]
        #  set the hyper-parameter
        # losses = [loss / batch_size / 1000 for loss in losses]

        return losses
