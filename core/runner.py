from data.dataset import FSSDataset
from core.backbone import Backbone
from eval.logger import Logger, AverageMeter
from eval.evaluation import Evaluator
from utils import commonutils as utils
import utils.segutils as segutils
import utils.crfhelper as crfutils
import core.contrastivehead as ctrutils
import core.denseaffinity as dautils
import torch

def set_args(_args):
    global args
    # _args should write benchmark, datapath, nshot, adapt-to, postprocessing, logpath, verbosity
    args = _args

    # then some more args are appended
    args.backbone = 'resnet50'
    args.nworker = 4 # tăng số luồng để chạy nhanh hơn
    args.bsz = 1 # the method works on a single task, hence bsz=1
    args.fold = 0


def makeDataloader():
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    return dataloader


def makeConfig():
    config = ctrutils.ContrastiveConfig()
    config.fitting.protoloss = False
    config.fitting.o_t_contr_proto_loss = True
    config.fitting.selfattentionloss = False
    config.fitting.keepvarloss = True
    config.fitting.symmetricloss = False
    config.fitting.q_nceloss = True
    config.fitting.s_nceloss = True
    config.fitting.num_epochs = 25
    config.fitting.lr = 1e-2
    config.fitting.debug = args.verbosity > 2
    config.model.out_channels = 64
    config.model.debug = args.verbosity > 0
    config.featext.fit_every_episode = False
    config.aug.blurkernelsize = [1]
    config.aug.n_transformed_imgs = 2
    config.aug.maxjitter = 0.0
    config.aug.maxangle = 0
    config.aug.maxscale = 1
    config.aug.maxshear = 20
    config.aug.apply_affine = True
    config.aug.debug = args.verbosity > 2
    return config


def makeFeatureMaker(dataset, config, device='cpu', randseed=2, feat_extr_method=None):
    utils.fix_randseed(randseed)
    if feat_extr_method is None:
        feat_extr_method = Backbone(args.backbone).to(device).extract_feats
    feat_maker = ctrutils.FeatureMaker(feat_extr_method, dataset.class_ids, config)
    utils.fix_randseed(randseed)
    feat_maker.norm_bb_feats = False
    return feat_maker


# Motivation of this class: Handle every task individually -> create an object for each task, example: see main.py
class SingleSampleEval:
    DEFAULT_FUSION_CANDIDATES = ('mean', 'high_suppressed', 'geomean', 'trimmed_mean', 'agreement')

    def __init__(self, batch, feat_maker):
        self.fusion_method = getattr(args, 'fusion_method', 'mean')
        self.fusion_params = self.make_fusion_params()
        self.damat_comp = dautils.DAMatComparison(self.fusion_method, self.fusion_params)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.feat_maker = feat_maker
        self.thresh_method = getattr(args, 'threshold_method', 'pred_mean')
        self.threshold_otsu_weight = getattr(args, 'threshold_otsu_weight', 0.5)
        self.post_proc_method = 'off'
        self.verbosity = args.verbosity
        self.selected_fusion_method = None

    def taskAdapt(self, detach=True):
        b = self.batch
        if self.device.type == 'cuda': b = utils.to_cuda(b)
        self.q_img, self.s_img, self.s_mask, self.class_id = b['query_img'], b['support_imgs'], b['support_masks'], b[
            'class_id'].item()
        self.task_adapted = self.feat_maker.taskAdapt(self.q_img, self.s_img, self.s_mask, self.class_id)

    @staticmethod
    def make_fusion_params():
        params = {
            'trim': getattr(args, 'fusion_trim', 3),
            'temperature': getattr(args, 'fusion_temperature', 10.0),
        }
        stage_weights = getattr(args, 'stage_weights', None)
        if stage_weights is not None:
            params['stage_weights'] = tuple(float(w) for w in stage_weights)
        return params

    def fusion_candidates(self):
        raw = getattr(args, 'fusion_candidates', None)
        if raw is None:
            return list(self.DEFAULT_FUSION_CANDIDATES)
        candidates = []
        for name in raw.split(','):
            name = name.strip()
            if not name or name in candidates:
                continue
            if name not in dautils.DAMatComparison.PURE_FUSION_METHODS:
                raise ValueError(
                    f'Unknown pseudo-gate fusion candidate {name!r}; '
                    f'choose from {sorted(dautils.DAMatComparison.PURE_FUSION_METHODS)}'
                )
            candidates.append(name)
        if not candidates:
            raise ValueError('--fusion-candidates must contain at least one method name')
        return candidates

    def pseudo_layer_predictions(self):
        """Per-layer predictions on the Algorithm-1 pseudo episode.

        The support image is the pseudo query and its augmentation (kept by
        the fitted contrastive adapter) is the pseudo support. The support
        mask is known, so candidate fusion methods can be compared without
        touching any query label.
        """
        s_feat = [f[0] for f in self.task_adapted[1] if f is not None]
        s_mask = self.s_mask[0]
        s_feataug, s_maskaug = crfutils.get_augmentations_and_features(self)
        # getTaskAdaptedFeats already removed the first l0 layers.
        return self.damat_comp.calc_q_pred_coarses(s_feat, s_feataug, s_maskaug, l0=0), s_mask

    def select_fusion_method(self):
        """Pick the fusion method with the best pseudo-episode IoU."""
        candidates = self.fusion_candidates()
        pseudo_layers, pseudo_targets = self.pseudo_layer_predictions()
        n_pseudo = pseudo_layers.shape[0]
        targets = pseudo_targets.float()

        scores = {}
        for candidate in candidates:
            candidate_scores = []
            for i in range(n_pseudo):
                fused = self.damat_comp.fuse_predictions(
                    pseudo_layers[i:i + 1], candidate, self.fusion_params
                )
                # Match the final path: upsample the fused map to mask
                # resolution, then threshold and score.
                fused = segutils.downsample_mask(fused, *targets.shape[-2:])
                thresh = self.thresh_method_fn(fused, targets[i:i + 1])
                pred_mask = (fused > thresh.reshape(-1, 1, 1)).float()
                candidate_scores.append(segutils.iou(pred_mask, targets[i:i + 1]))
            scores[candidate] = sum(candidate_scores) / len(candidate_scores)

        best = max(candidates, key=lambda name: scores[name])
        if self.verbosity > 0:
            printable = ', '.join(f'{name}={value:.4f}' for name, value in scores.items())
            print(f'[pseudo-gate] selected {best!r} from pseudo-episode IoU: {printable}')
        return best

    def thresh_method_fn(self, fused_pred, s_masks):
        return segutils.calcthresh(
            fused_pred, s_masks, self.thresh_method, self.threshold_otsu_weight
        )

    def compare_feats(self):
        if self.task_adapted is None:
            print("error, do task adaption first")
            return None
        fusion_method = self.fusion_method
        if fusion_method == 'pseudo_gate':
            fusion_method = self.select_fusion_method()
            self.selected_fusion_method = fusion_method
        self.logit_mask = self.damat_comp.forward(
            self.task_adapted[0], self.task_adapted[1], self.s_mask,
            fusion_method=fusion_method,
        )
        return self.logit_mask

    def threshold(self, method=None):
        if self.logit_mask is None:
            print("error, calculate logit mask first (do forward pass)")
        if method is None:
            method = self.thresh_method
        self.thresh = segutils.calcthresh(
            self.logit_mask, self.s_mask, method, self.threshold_otsu_weight
        )
        self.pred_mask = (self.logit_mask > self.thresh.reshape(-1, 1, 1)).float()
        return self.thresh, self.pred_mask

    def postprocess(self):
        if self.post_proc_method == 'off':
            return self.pred_mask
        elif self.post_proc_method == 'always':
            return crfutils.apply_crf(
                self.q_img, self.logit_mask,
                segutils.thresh_fn(self.thresh_method, self.threshold_otsu_weight)
            ).to(self.device)
        elif self.post_proc_method == 'dynamic':
            if crfutils.crf_is_good(self):
                return crfutils.apply_crf(
                    self.q_img, self.logit_mask,
                    segutils.thresh_fn(self.thresh_method, self.threshold_otsu_weight)
                ).to(self.device)
            return self.pred_mask
        elif self.post_proc_method == 'guided' or self.post_proc_method.startswith('guided-morph-'):
            refined = segutils.guided_filter(self.q_img, self.logit_mask)
            refined_thresh = self.thresh_method_fn(refined, self.s_mask)
            self.thresh = refined_thresh
            refined_mask = (refined > refined_thresh[:, None, None]).float()
            if self.post_proc_method == 'guided':
                return refined_mask
            operation = self.post_proc_method[len('guided-morph-'):].replace('-', '_')
            return segutils.morphological_filter(refined_mask, operation)
        elif self.post_proc_method in {'morph-open', 'morph-close', 'morph-open-close'}:
            operation = self.post_proc_method[len('morph-'):].replace('-', '_')
            return segutils.morphological_filter(self.pred_mask, operation)
        else:
            raise ValueError(f'Unknown post-processing method: {self.post_proc_method!r}')

    # this method calls above components sequentially
    def forward(self):
        self.taskAdapt()

        self.logit_mask = self.compare_feats()

        self.thresh, self.pred_mask = self.threshold()

        self.pred_mask = self.postprocess()

        return self.logit_mask, self.pred_mask

    def calc_metrics(self):
        # assert torch.logical_or(self.logit_mask<0, self.logit_mask>1).sum()==0, display(tensor_table(logit_mask=self.logit_mask))
        self.area_inter, self.area_union = Evaluator.classify_prediction(self.pred_mask, self.batch)
        self.fgratio_pred = self.pred_mask.float().mean()
        self.fgratio_gt = self.batch['query_mask'].float().mean()
        return self.area_inter[1] / self.area_union[1]  # fg-iou

    def plots(self):
        display(pilImageRow(norm(self.logit_mask[0]), (self.logit_mask[0] > self.thresh).float(), self.pred_mask,
                            self.batch['query_mask'][:1], norm(self.q_img[0]), norm(self.s_img[0, 0])))
        display(segutils.tensor_table(probs=self.logit_mask))

        print('s_mask.mean, pred_mask.mean, thresh:', self.s_mask.mean().item(), self.logit_mask.mean().item(),
              self.thresh.item())


class AverageMeterWrapper:
    def __init__(self, dataloader, device='cpu', initlogger=True):
        if initlogger: Logger.initialize(args, training=False)
        self.average_meter = AverageMeter(dataloader.dataset, device)
        self.device = device
        self.dataloader = dataloader
        self.write_batch_idx = 50

    def update(self, sseval):
        self.average_meter.update(sseval.area_inter, sseval.area_union, torch.tensor(sseval.class_id).to(self.device),
                                  loss=None)

    def update_manual(self, area_inter, area_union, class_id):
        if isinstance(class_id, int): class_id = torch.tensor(class_id).to(self.device)
        self.average_meter.update(area_inter, area_union, class_id, loss=None)

    def write(self, i):
        self.average_meter.write_process(i, len(self.dataloader), 0, self.write_batch_idx)

