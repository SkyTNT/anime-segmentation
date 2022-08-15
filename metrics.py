import os

import torch


def f1_torch(pred, gt):
    pred = pred.float().view(pred.shape[0], -1)
    gt = gt.float().view(gt.shape[0], -1)
    tp = torch.sum(pred * gt, dim=1)
    precision = tp / (pred.sum(dim=1) + 0.0001)
    recall = tp / (gt.sum(dim=1) + 0.0001)
    f1 = (1 + 0.3) * precision * recall / (0.3 * precision + recall + 0.0001)
    return precision, recall, f1
