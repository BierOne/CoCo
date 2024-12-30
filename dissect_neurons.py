# New-style dissection experiment code.
import sys
import torch, torchvision, argparse, os, shutil, inspect, json, numpy, random
from collections import defaultdict
from torch.utils.data import ConcatDataset
from easydict import EasyDict as edict

sys.path.append(os.path.expanduser("~/dg/dissect"))
from netdissect import pbar, nethook, renormalize, pidfile, zdataset
from netdissect import upsample, tally, imgviz, imgsave, bargraph, parallelfolder
from experiment import setting
import netdissect
torch.backends.cudnn.benchmark = True

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

### --- COMMAND FOR RUNNING ---
# export CUDA_HOME=/usr/local/cuda-11.1/
# python -m experiment.dissect_experiment_ours --model resnet18
def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--steps', type=int, default=0,
        help='Number of steps for the checkpoint; if None, then dissect all checkpoints')
    aa('--seg', choices=['net', 'netp', 'netq', 'netpq', 'netpqc', 'netpqxc'],
            default='netpqc')
    aa('--compute_seg', action='store_true')
    aa('--data_dir', type=str, default="/dg/data/domainbed")
    aa('--layer_name', type=str, default="featurizer.network.layer4")
    aa('--model_dir', type=str, default="results/resnet50/PACS_ERM_0")
    aa('--topk', "-k", type=int, default=5)
    aa('--quantile', type=float, default=0.01)
    aa('--miniou', type=float, default=0.04)
    aa('--thumbsize', type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    layername = args.layer_name

    ################### --- Modify Here --- ###################
    name = "model_step{}.pkl".format(args.steps) if args.steps >=0 else "model.pkl"
    model_path = os.path.join(args.model_dir, name)

    logs = torch.load(model_path)
    hparams = logs['model_hparams']
    m_args = edict(logs['args'])

    print('Loaded Model Args from {}:'.format(model_path))
    for k, v in sorted(vars(m_args).items()):
        print('\t{}: {}'.format(k, v))

    hparams['data_augmentation'] = False  # for better visualization
    dataset = vars(datasets)[m_args.dataset](m_args.data_dir,
                                             m_args.test_envs, hparams)
    algorithm_class = algorithms.get_algorithm_class(m_args.algorithm)
    model = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(m_args.test_envs), hparams)

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*m_args.holdout_fraction),
            misc.seed_hash(m_args.trial_seed, env_i))

        if env_i in m_args.test_envs:
            _, in_ = misc.split_dataset(in_,
                int(len(in_)*m_args.uda_holdout_fraction),
                misc.seed_hash(m_args.trial_seed, env_i))
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
    dataset = ConcatDataset([env for i, (env, env_weights) in enumerate(in_splits)
                             if i not in m_args.test_envs])

    model.load_state_dict(logs['model_dict'])
    model = nethook.InstrumentedModel(model).cuda().eval()

    ################### --- Modify Here --- ###################
    resdir = '~/dg/visualizations/neuron_state/%s/%s/env%s' % (m_args.dataset, m_args.algorithm, m_args.test_envs[0])
    if args.steps >=0:
        resdir += '-s%d' % (args.steps)
    if args.layer_name is not None:
        resdir += '-' + args.layer_name.split('.')[-1]
    if args.quantile != 0.005:
        resdir += ('-%g' % (args.quantile * 1000))
    if args.thumbsize != 100:
        resdir += ('-t%d' % (args.thumbsize))
    resdir = os.path.expanduser(resdir)
    if os.path.isdir(resdir):
        print("remove the old output dir:", resdir)
        shutil.rmtree(resdir)

    resfile = pidfile.exclusive_dirfn(resdir)


    model.retain_layer(layername)
    upfn = make_upfn(args, dataset, model, layername)
    sample_size = len(dataset)
    percent_level = 1.0 - args.quantile
    iou_threshold = args.miniou
    image_row_width = args.topk
    torch.set_grad_enabled(False)

    # Tally rq.np (representation quantile, unconditional).
    pbar.descnext('rq')
    def compute_samples(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
    rq = tally.tally_quantile(compute_samples, dataset,
                              sample_size=sample_size,
                              r=8192,
                              num_workers=30,
                              pin_memory=True,
                              batch_size=100,
                              cachefile=resfile('rq.npz'))

    # Create visualizations - first we need to know the topk
    pbar.descnext('topk')
    def compute_image_max(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.max(2)[0]
        return acts
    topk = tally.tally_topk(compute_image_max, dataset, sample_size=sample_size,
            batch_size=100, num_workers=30, pin_memory=True,
            cachefile=resfile('topk.npz'))

    # Visualize top-activating patches of top-activatin images.
    pbar.descnext('unit_images')
    image_size, image_source = None, None
    image_source = dataset
    iv = imgviz.ImageVisualizer((args.thumbsize, args.thumbsize),
        image_size=image_size,
        source=dataset,
        quantiles=rq,
        level=rq.quantiles(percent_level))
    def compute_acts(data_batch, *ignored_class):
        data_batch = data_batch.cuda()
        out_batch = model(data_batch)
        acts_batch = model.retained_layer(layername)
        return (acts_batch, data_batch)


    unit_images = iv.masked_images_for_topk(
        compute_acts, dataset, topk,
        k=image_row_width, num_workers=30, pin_memory=True,
        cachefile=resfile('top%dimages.npz' % image_row_width))
    pbar.descnext('saving images')
    imgsave.save_image_set(unit_images, resfile('image/unit%d.jpg'),
                           sourcefile=resfile('top%dimages.npz' % image_row_width))

    ########### visualize original image ##############
    # unit_images = iv.org_images_for_topk(
    #         compute_acts, dataset, topk,
    #         k=image_row_width, num_workers=30, pin_memory=True,
    #         cachefile=resfile('org_top%dimages.npz' % image_row_width))
    # pbar.descnext('saving images')
    # imgsave.save_image_set(unit_images, resfile('org_image/unit%d.jpg'),
    #         sourcefile=resfile('org_top%dimages.npz' % image_row_width))
    ######################   END  #####################



    # Compute IoU agreement between segmentation labels and every unit
    # Grab the 99th percentile, and tally conditional means at that level.
    if args.compute_seg:
        level_at_99 = rq.quantiles(percent_level).cuda()[None,:,None,None]

        segmodel, seglabels, segcatlabels = setting.load_segmenter(args.seg)
        renorm = renormalize.renormalizer(dataset, target='zc')
        def compute_conditional_indicator(batch, *args):
            data_batch = batch.cuda()
            out_batch = model(data_batch)
            image_batch = renorm(data_batch)
            seg = segmodel.segment_batch(image_batch, downsample=4)
            acts = model.retained_layer(layername)
            hacts = upfn(acts)
            iacts = (hacts > level_at_99).float() # indicator
            return tally.conditional_samples(iacts, seg)

        pbar.descnext('condi99')
        condi99 = tally.tally_conditional_mean(compute_conditional_indicator,
                dataset, sample_size=sample_size,
                num_workers=3, pin_memory=True,
                cachefile=resfile('condi99.npz'))

        # Now summarize the iou stats and graph the units
        iou_99 = tally.iou_from_conditional_indicator_mean(condi99)
        unit_label_99 = [
                (concept.item(), seglabels[concept],
                    segcatlabels[concept], bestiou.item())
                for (bestiou, concept) in zip(*iou_99.max(0))]
        labelcat_list = [labelcat
                for concept, label, labelcat, iou in unit_label_99
                if iou > iou_threshold]
        save_conceptcat_graph(resfile('concepts_99.svg'), labelcat_list)
        dump_json_file(resfile('report.json'), dict(
                header=dict(
                    name='%s %s %s' % (m_args.algorithm, m_args.dataset, args.seg),
                    image='concepts_99.svg'),
                units=[
                    dict(image='image/unit%d.jpg' % u,
                        unit=u, iou=iou, label=label, cat=labelcat[1])
                    for u, (concept, label, labelcat, iou)
                    in enumerate(unit_label_99)])
                )
        copy_static_file('report.html', resfile('+report.html'))
        resfile.done()
    print('Done!')

def make_upfn(args, dataset, model, layername):
    '''Creates an upsampling function.'''
    convs, data_shape = None, None
    # Probe the data shape
    _ = model(dataset[0][0][None,...].cuda())
    data_shape = model.retained_layer(layername).shape[2:]
    pbar.print('upsampling from data_shape', tuple(data_shape))
    upfn = upsample.upsampler(
            (56, 56),
            data_shape=data_shape,
            source=dataset,
            convolutions=convs)
    return upfn


def graph_conceptcatlist(conceptcatlist, **kwargs):
    count = defaultdict(int)
    catcount = defaultdict(int)
    for c in conceptcatlist:
        count[c] += 1
    for c in count.keys():
        catcount[c[1]] += 1
    cats = ['object', 'part', 'material', 'texture', 'color']
    catorder = dict((c, i) for i, c in enumerate(cats))
    sorted_labels = sorted(count.keys(),
        key=lambda x: (catorder[x[1]], -count[x]))
    sorted_labels
    return bargraph.make_svg_bargraph(
        [label for label, cat in sorted_labels],
        [count[k] for k in sorted_labels],
        [(c, catcount[c]) for c in cats], **kwargs)

def save_conceptcat_graph(filename, conceptcatlist):
    svg = graph_conceptcatlist(conceptcatlist, barheight=80, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)


class FloatEncoder(json.JSONEncoder):
    def __init__(self, nan_str='"NaN"', **kwargs):
        super(FloatEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def iterencode(self, o, _one_shot=False):
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring
        def floatstr(o, allow_nan=self.allow_nan,
                _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY,
                nan_str=self.nan_str):
            if o != o:
                text = nan_str
            elif o == _inf:
                text = '"Infinity"'
            elif o == _neginf:
                text = '"-Infinity"'
            else:
                return repr(o)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))
            return text

        _iterencode = json.encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

def dump_json_file(target, data):
    with open(target, 'w') as f:
        json.dump(data, f, indent=1, cls=FloatEncoder)

def copy_static_file(source, target):
    sourcefile = os.path.join(
            os.path.dirname(inspect.getfile(netdissect)), source)
    shutil.copy(sourcefile, target)

if __name__ == '__main__':
    main()

