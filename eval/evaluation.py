r""" Evaluate mask prediction """
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    ignore_index = 255

    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch (vectorized)
        pred = pred_mask.view(pred_mask.shape[0], -1)
        gt = gt_mask.view(gt_mask.shape[0], -1)

        valid = (gt != cls.ignore_index)

        inter_0 = ((pred == 0) & (gt == 0) & valid).sum(dim=1)
        inter_1 = ((pred == 1) & (gt == 1) & valid).sum(dim=1)
        area_inter = torch.stack([inter_0, inter_1], dim=0)

        pred_0 = ((pred == 0) & valid).sum(dim=1)
        pred_1 = ((pred == 1) & valid).sum(dim=1)
        area_pred = torch.stack([pred_0, pred_1], dim=0)

        gt_0 = ((gt == 0) & valid).sum(dim=1)
        gt_1 = ((gt == 1) & valid).sum(dim=1)
        area_gt = torch.stack([gt_0, gt_1], dim=0)

        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union
