from core import runner
from core.denseaffinity import DAMatComparison
import torch
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Adapt Before Comparison - A New Perspective on Cross-Domain Few-Shot Segmentation')

    parser.add_argument('--benchmark', type=str, choices=['fss', 'deepglobe', 'lung', 'isic', 'fss', 'suim'])
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--adapt-to', type=str, default='first-episode', choices=['first-episode', 'every-episode'])
    parser.add_argument(
        '--fusion-method', type=str, default='mean',
        choices=sorted(DAMatComparison.FUSION_METHODS),
        help='How to combine the 13 ResNet50 layer predictions. '
             'high_suppressed downweights the coarse high stage and is tuned for ISIC; '
             'pseudo_gate picks a method per episode on the Algorithm-1 pseudo episode.',
    )
    parser.add_argument(
        '--stage-weights', type=float, nargs=3, default=None, metavar=('WL', 'WM', 'WH'),
        help='Stage weights for weighted_stage / weighted_stage_geomean. '
             'The ISIC group-A winner is 1 1 0.25 (same ratio as A5_w2_2_0.5).',
    )
    parser.add_argument(
        '--fusion-trim', type=int, default=3,
        help='Layers dropped from each side of the sorted layer predictions for trimmed_mean',
    )
    parser.add_argument(
        '--fusion-temperature', type=float, default=10.0,
        help='Softmax sharpening for agreement weighting',
    )
    parser.add_argument(
        '--fusion-candidates', type=str, default=None,
        help='Comma-separated candidates for pseudo_gate '
             '(default: mean,high_suppressed,geomean,trimmed_mean,agreement)',
    )
    parser.add_argument(
        '--threshold-method', type=str, default='pred_mean',
        choices=['otsus', 'mean', 'pred_mean'],
        help='Threshold for converting foreground probabilities to a binary mask',
    )
    parser.add_argument(
        '--postprocessing', type=str, default='off',
        choices=['off', 'dynamic', 'always', 'guided', 'morph-open', 'morph-close', 'morph-open-close'],
        help='Optional refinement after thresholding; dynamic/always use DenseCRF',
    )
    parser.add_argument('--logpath', type=str, default='./logs')
    parser.add_argument('--verbosity', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_opts()
    runner.set_args(args)

    dataloader = runner.makeDataloader()
    config = runner.makeConfig()
    config.featext.fit_every_episode = args.adapt_to == 'every-episode'
    feat_maker = runner.makeFeatureMaker(dataloader.dataset, config, device=device)
    average_meter = runner.AverageMeterWrapper(dataloader, device)

    for idx, batch in enumerate(dataloader):
        sseval = runner.SingleSampleEval(batch, feat_maker)
        sseval.post_proc_method = args.postprocessing
        sseval.forward()
        sseval.calc_metrics()
        average_meter.update(sseval)
        average_meter.write(idx)
    print('Result m|FB:', average_meter.average_meter.compute_iou())
