import torch
import torch.nn.functional as F
import math
from utils import segutils


def buildHyperCol(feat_pyram):
    # concatenate along channel dim
    # upsample spatial size to largest feat vol space available
    target_size = feat_pyram[0].shape[-2:]
    upsampled = [feat_pyram[0]]  # first layer already at target size
    for layer in feat_pyram[1:]:
        upsampled.append(F.interpolate(layer, size=target_size, mode='bilinear', align_corners=False))
    return torch.cat(upsampled, dim=1)


# accepts both:
# s_feat_vol: [bsz,k,c,h,w]->[bsz,c,h,w*k]
# s_mask: [bsz,k,h,w]->[bsz,h,w*k]
def paste_supports_together(supports):
    return torch.cat(supports.unbind(dim=1), dim=-1)


# Attention regular:
# 1. Dot product
# 2. Divide by square root of key length (#nchannels)
# 3. Softmax
# 4. Multiply with V (mask)

def buildDenseAffinityMat(qfeat_volume, sfeat_volume, softmax_arg2=True):  # bsz,C,H,W
    bsz, C = qfeat_volume.shape[:2]
    # Fast bmm without permute memory copy:
    # q: (bsz, C, H*Wq) -> transpose -> (bsz, H*Wq, C)
    # s: (bsz, C, H*Ws)
    dense_affinity_mat = torch.bmm(qfeat_volume.flatten(2).transpose(1, 2),
                                   sfeat_volume.flatten(2))
    if softmax_arg2 is False: return dense_affinity_mat
    dense_affinity_mat_softmax = (dense_affinity_mat / math.sqrt(C)).softmax(
        dim=-1)  # each query pixel's affinities sum up to 1 over support pxls
    return dense_affinity_mat_softmax


# filter with support mask following DAM
def filterDenseAffinityMap(dense_affinity_mat, downsampled_smask):
    # for each query pixel, aggregate all correlations where the support mask ==1
    # [px,px][px,1]=[px,1]
    bsz, HWq, HWs = dense_affinity_mat.shape
    q_coarse = torch.bmm(dense_affinity_mat, downsampled_smask.view(bsz, HWs, 1))
    return q_coarse.view(bsz, HWq)


def upsample(volume, h, w):
    return F.interpolate(volume, size=(h, w), mode='bilinear', align_corners=False)

class DAMatComparison:

    def algo_mean(self, q_pred_coarses_t, s_mask=None):
        return q_pred_coarses_t.mean(1)

    def calc_q_pred_coarses(self, q_feat_t, s_feat_t, s_mask, l0=3):
        q_pred_coarses = []
        h0, w0 = q_feat_t[l0].shape[-2:]
        with torch.no_grad():
            for (qft, sft) in zip(q_feat_t[l0:], s_feat_t[l0:]):
                qft, sft = qft.detach(), sft.detach()
                bsz, c, hq, wq = qft.shape
                hs, ws = sft.shape[-2:]

                sft_row = torch.cat(sft.unbind(1), -1)  # bsz,k,c,h,w -> bsz,c,h,w*k
                k = s_mask.shape[1]
                smasks_down = F.interpolate(s_mask.view(bsz * k, 1, *s_mask.shape[-2:]), size=(hs, ws),
                                            mode='bilinear', align_corners=False).view(bsz, k, hs, ws)
                smask_row = torch.cat(smasks_down.unbind(1), -1)

                damat = buildDenseAffinityMat(qft, sft_row)
                filtered = filterDenseAffinityMap(damat, smask_row)
                q_pred_coarse = upsample(filtered.view(bsz, 1, hq, wq), h0, w0).squeeze(1)
                q_pred_coarses.append(q_pred_coarse)
        return torch.stack(q_pred_coarses, dim=1)

    def forward(self, q_feat_t, s_feat_t, s_mask, upsample=True, debug=False):
        q_pred_coarses_t = self.calc_q_pred_coarses(q_feat_t, s_feat_t, s_mask)

        if debug: display(segutils.pilImageRow(*q_pred_coarses_t.unbind(1), q_pred_coarses_t.mean(1)))

        # select the algorithm
        postprocessing_algorithm = self.algo_mean
        # do the postprocessing
        logit_mask = postprocessing_algorithm(q_pred_coarses_t, s_mask)
        if upsample:  # if query and support have different shape, then you must do upsampling yourself afterwards
            logit_mask = segutils.downsample_mask(logit_mask, *s_mask.shape[-2:])

        return logit_mask