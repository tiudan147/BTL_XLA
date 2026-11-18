import torch
import torch.nn.functional as F
import math
from utils import segutils


def buildHyperCol(feat_pyram):
    # concatenate along channel dim
    # upsample spatial size to largest feat vol space available
    target_size = feat_pyram[0].shape[-2:]
    upsampled = []
    for layer in feat_pyram:
        # if idx < self.stack_ids[0]: continue
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
    qfeat_volume, sfeat_volume = qfeat_volume.permute(0, 2, 3, 1), sfeat_volume.permute(0, 2, 3, 1)
    bsz, H, Wq, C = qfeat_volume.shape
    Ws = sfeat_volume.shape[2]
    # [px,C][C,px]=[px,px]
    dense_affinity_mat = torch.matmul(qfeat_volume.view(bsz, H * Wq, C),
                                      sfeat_volume.view(bsz, H * Ws, C).transpose(1, 2))
    if softmax_arg2 is False: return dense_affinity_mat
    dense_affinity_mat_softmax = (dense_affinity_mat / math.sqrt(C)).softmax(
        dim=-1)  # each query pixel's affinities sum up to 1 over support pxls
    return dense_affinity_mat_softmax


# filter with support mask following DAM
def filterDenseAffinityMap(dense_affinity_mat, downsampled_smask):
    # for each query pixel, aggregate all correlations where the support mask ==1
    # [px,px][px,1]=[px,1]
    bsz, HWq, HWs = dense_affinity_mat.shape
    # let mean(V)=1 -> sum(V)=len(V) -> d_mask / mean(d_mask)
    # downsampled_smask_norm = downsampled_smask / downsampled_smask.mean()
    q_coarse = torch.matmul(dense_affinity_mat, downsampled_smask.view(bsz, HWs, 1))
    return q_coarse.view(bsz, HWq)


def upsample(volume, h, w):
    return F.interpolate(volume, size=(h, w), mode='bilinear', align_corners=False)

class DAMatComparison:

    FUSION_METHODS = {
        'mean', 'geomean', 'low', 'mid', 'high', 'stage_uniform', 'low_biased', 'high_biased',
        # Group A additions, focused on the ISIC domain.
        'high_suppressed', 'weighted_stage', 'weighted_stage_geomean', 'trimmed_mean', 'agreement',
        # Per-episode selection between fusion candidates (resolved in
        # SingleSampleEval before fusion, never passed to fuse_predictions).
        'pseudo_gate',
    }
    PURE_FUSION_METHODS = FUSION_METHODS - {'pseudo_gate'}
    STAGE_SLICES = (slice(0, 4), slice(4, 10), slice(10, 13))
    STAGE_WEIGHTS = {
        'stage_uniform': (1 / 3, 1 / 3, 1 / 3),
        'low_biased': (0.5, 0.3, 0.2),
        'high_biased': (0.2, 0.3, 0.5),
        # Empirically the best group-A setting on ISIC (A5_w2_2_0.5 in
        # docs/exp.md): keep mid/low, strongly suppress the coarse high stage.
        'high_suppressed': (1.0, 1.0, 0.25),
    }

    def __init__(self, fusion_method='mean', fusion_params=None):
        self.fusion_params = dict(fusion_params or {})
        self.set_fusion_method(fusion_method)

    def set_fusion_method(self, fusion_method):
        if fusion_method not in self.FUSION_METHODS:
            raise ValueError(
                f"Unknown fusion method {fusion_method!r}; "
                f"choose from {sorted(self.FUSION_METHODS)}"
            )
        self.fusion_method = fusion_method

    @staticmethod
    def _check_layers(layers):
        if layers.ndim != 4:
            raise ValueError('Layer predictions must have shape [B, layers, H, W]')
        if not torch.isfinite(layers).all():
            raise ValueError('Layer predictions contain non-finite values')
        return layers

    @classmethod
    def _stage_maps(cls, layers):
        return torch.stack(
            [layers[:, stage].mean(dim=1) for stage in cls.STAGE_SLICES],
            dim=1,
        )

    @classmethod
    def _stage_weights(cls, layers, params, default=(1.0, 1.0, 1.0)):
        raw = params.get('stage_weights', params.get('weights', default))
        if len(raw) != 3:
            raise ValueError('stage weights must contain exactly three values (L, M, H)')
        weights = layers.new_tensor([float(w) for w in raw])
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(f'Invalid stage weights {raw!r}; expected three non-negative values with a positive sum')
        return weights

    @classmethod
    def fuse_predictions(cls, q_pred_coarses_t, fusion_method='mean', fusion_params=None):
        params = dict(fusion_params or {})
        if fusion_method not in cls.PURE_FUSION_METHODS:
            raise ValueError(
                f"Unknown fusion method {fusion_method!r}; "
                f"choose from {sorted(cls.PURE_FUSION_METHODS)}"
            )
        q_pred_coarses_t = cls._check_layers(q_pred_coarses_t)
        if fusion_method == 'mean':
            return q_pred_coarses_t.mean(dim=1)
        if fusion_method == 'geomean':
            # Geometric mean in log space; clamp keeps zero-scored pixels finite.
            return torch.exp(torch.log(q_pred_coarses_t.clamp_min(1e-6)).mean(dim=1))
        if fusion_method == 'trimmed_mean':
            trim = int(params.get('trim', 3))
            n_layers = q_pred_coarses_t.shape[1]
            if trim < 0 or 2 * trim >= n_layers:
                raise ValueError(
                    f'Trimmed mean cannot drop {trim} layers from each side of {n_layers} layers'
                )
            ordered = q_pred_coarses_t.sort(dim=1).values
            return ordered[:, trim:n_layers - trim].mean(dim=1)
        if q_pred_coarses_t.shape[1] != 13:
            raise ValueError(
                f"Fusion method {fusion_method!r} requires 13 layer predictions; "
                f"got {q_pred_coarses_t.shape[1]}"
            )

        stage_predictions = cls._stage_maps(q_pred_coarses_t)
        if fusion_method == 'low':
            return stage_predictions[:, 0]
        if fusion_method == 'mid':
            return stage_predictions[:, 1]
        if fusion_method == 'high':
            return stage_predictions[:, 2]
        if fusion_method == 'weighted_stage':
            weights = cls._stage_weights(q_pred_coarses_t, params)
            return (stage_predictions * weights.view(1, 3, 1, 1)).sum(dim=1) / weights.sum()
        if fusion_method == 'weighted_stage_geomean':
            weights = cls._stage_weights(q_pred_coarses_t, params)
            log_stages = torch.log(stage_predictions.clamp_min(1e-6))
            return torch.exp((log_stages * weights.view(1, 3, 1, 1)).sum(dim=1) / weights.sum())
        if fusion_method == 'agreement':
            # Per-episode stage reliability: each stage map is centred and
            # normalized independently, then compared with the robust (median)
            # stage consensus. A coarse or noisy stage that disagrees with the
            # other stages is downweighted; raw score scale cannot bias the
            # weights because every stage is normalized first.
            stage_flat = stage_predictions.flatten(2)
            stage_flat = stage_flat - stage_flat.mean(dim=-1, keepdim=True)
            stage_flat = stage_flat / stage_flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            consensus_flat = stage_flat.median(dim=1).values
            consensus_flat = consensus_flat / consensus_flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            similarity = (stage_flat * consensus_flat.unsqueeze(1)).sum(dim=-1)
            temperature = float(params.get('temperature', 10.0))
            weights = torch.softmax(similarity * temperature, dim=1)
            return (stage_predictions * weights.view(-1, 3, 1, 1)).sum(dim=1)

        weights = q_pred_coarses_t.new_tensor(cls.STAGE_WEIGHTS[fusion_method])
        return (stage_predictions * weights.view(1, 3, 1, 1)).sum(dim=1) / weights.sum()

    def algo_mean(self, q_pred_coarses_t, s_mask=None):
        return q_pred_coarses_t.mean(1)

    def calc_q_pred_coarses(self, q_feat_t, s_feat_t, s_mask, l0=3):
        if len(q_feat_t) != len(s_feat_t):
            raise ValueError('Query and support must provide the same number of feature maps')
        q_pred_coarses = []
        h0, w0 = q_feat_t[l0].shape[-2:]
        for (qft, sft) in zip(q_feat_t[l0:], s_feat_t[l0:]):
            qft, sft = qft.detach(), sft.detach()
            bsz, c, hq, wq = qft.shape
            hs, ws = sft.shape[-2:]

            sft_row = torch.cat(sft.unbind(1), -1)  # bsz,k,c,h,w -> bsz,c,h,w*k
            smasks_downsampled = [segutils.downsample_mask(m, hs, ws) for m in s_mask.unbind(1)]
            smask_row = torch.cat(smasks_downsampled, -1)

            damat = buildDenseAffinityMat(qft, sft_row)
            filtered = filterDenseAffinityMap(damat, smask_row)
            q_pred_coarse = upsample(filtered.view(bsz, 1, hq, wq), h0, w0).squeeze(1)
            q_pred_coarses.append(q_pred_coarse)
        return torch.stack(q_pred_coarses, dim=1)

    def forward(self, q_feat_t, s_feat_t, s_mask, upsample=True, debug=False,
                fusion_method=None, fusion_params=None, l0=3):
        q_pred_coarses_t = self.calc_q_pred_coarses(q_feat_t, s_feat_t, s_mask, l0=l0)

        if debug: display(segutils.pilImageRow(*q_pred_coarses_t.unbind(1), q_pred_coarses_t.mean(1)))

        if fusion_method is None:
            fusion_method = self.fusion_method
        if fusion_params is None:
            fusion_params = self.fusion_params
        logit_mask = self.fuse_predictions(q_pred_coarses_t, fusion_method, fusion_params)
        if upsample:  # if query and support have different shape, then you must do upsampling yourself afterwards
            logit_mask = segutils.downsample_mask(logit_mask, *s_mask.shape[-2:])

        return logit_mask
