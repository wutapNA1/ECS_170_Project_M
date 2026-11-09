'''
Runs 3-3 (MNIST), 3-4 (ORL, CIFAR) and 3-5 (configuration study).
Expected project layout (paths are resolved from this file, so it can be run from anywhere):
    <project>/local_code/stage_3_code/ Dataset_Loader.py, Method_CNN*.py, Evaluate_Metrics.py
    <project>/script/stage_3_script/  script_cnn.py   (this file)
    <project>/data/stage_3_data/      MNIST, ORL, CIFAR
    <project>/result/stage_3_result/  created automatically
Usage:  python3 script_cnn.py [--ablation | --full_ablation] [--configs ...] [--quick]
'''
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)                                    # lets local_code.base_class be imported
sys.path.insert(0, os.path.join(ROOT, 'local_code', 'stage_3_code'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Dataset_Loader import Dataset_Loader
from Method_CNN_MNIST import Method_CNN_MNIST
from Method_CNN_ORL import Method_CNN_ORL
from Method_CNN_CIFAR import Method_CNN_CIFAR
from Evaluate_Metrics import Evaluate_Metrics

FILES = {'MNIST': 'MNIST', 'ORL': 'ORL', 'CIFAR': 'CIFAR'}
METHODS = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}

# 3-5: one-factor-at-a-time changes relative to each dataset's baseline
ABLATIONS = {
    'baseline': {},
    'depth_1': dict(conv_channels=[32]),
    'depth_4': dict(conv_channels=[32, 64, 128, 128]),
    'kernel_5': dict(kernel_size=5, padding=2),
    'kernel_7': dict(kernel_size=7, padding=3),
    'no_padding': dict(padding=0),
    'stride_2': dict(stride=2),
    'avg_pool': dict(pool='avg'),
    'no_pool': dict(pool=None),
    'hidden_64': dict(hidden_dims=[64]),
    'hidden_512x2': dict(hidden_dims=[512, 512]),
    'loss_mse': dict(loss='mse'),
    'loss_smooth': dict(loss='smooth'),
    'no_bn': dict(batch_norm=False),
}


# subset used for 3-5 (covers depth, kernel size, pooling, hidden size and loss function)
SUBSET = ['baseline', 'depth_1', 'depth_4', 'kernel_5', 'avg_pool', 'hidden_64', 'loss_mse']


def plot_curves(h, title, path):
    ep = range(1, len(h['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, h['train_loss'], label='train'); ax[0].plot(ep, h['test_loss'], label='test')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].legend(); ax[0].set_title(title + ' loss')
    ax[1].plot(ep, h['train_acc'], label='train'); ax[1].plot(ep, h['test_acc'], label='test')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('accuracy'); ax[1].legend(); ax[1].set_title(title + ' accuracy')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_one(ds_name, data, cfg, tag, out_dir):
    model = METHODS[ds_name](**cfg)
    model.data = data
    out = model.run()
    ev = Evaluate_Metrics('metrics', '')
    ev.data = out
    scores = ev.evaluate()
    plot_curves(model.history, '%s %s' % (ds_name, tag), os.path.join(out_dir, '%s_%s_curves.png' % (ds_name, tag)))
    print('>>', ds_name, tag, scores)
    return {'dataset': ds_name, 'config': tag, **scores,
            'params': sum(p.numel() for p in model.parameters()), 'history': model.history}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data', 'stage_3_data') + os.sep)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'result', 'stage_3_result') + os.sep)
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='default: MNIST ORL CIFAR, or MNIST CIFAR with --ablation')
    ap.add_argument('--ablation', action='store_true',
                    help='3-5 study on the SUBSET of configurations (MNIST and CIFAR by default)')
    ap.add_argument('--full_ablation', action='store_true',
                    help='3-5 study on every configuration in ABLATIONS (slow)')
    ap.add_argument('--configs', nargs='+', default=None,
                    help='only these configuration names, e.g. --configs baseline depth_1 kernel_5')
    ap.add_argument('--quick', action='store_true', help='3 epochs, for smoke tests')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    study = a.ablation or a.full_ablation or bool(a.configs)
    if a.datasets is None:
        a.datasets = ['MNIST', 'CIFAR'] if study else ['MNIST', 'ORL', 'CIFAR']
    suffix = '_ablation' if study else '_baseline'
    json_path = os.path.join(a.out_dir, 'results%s.json' % suffix)
    csv_path = os.path.join(a.out_dir, 'results%s.csv' % suffix)

    # resume: keep earlier finished runs and skip them
    results = []
    if os.path.exists(json_path) and not a.quick:
        with open(json_path) as f:
            results = json.load(f)
    done = {(r['dataset'], r['config']) for r in results}

    def save_results():
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=1)
        with open(csv_path, 'w') as f:
            f.write('dataset,config,accuracy,precision,recall,f1,params\n')
            for r in results:
                f.write('%s,%s,%.4f,%.4f,%.4f,%.4f,%d\n' % (r['dataset'], r['config'], r['accuracy'],
                        r['precision'], r['recall'], r['f1'], r['params']))

    for name in a.datasets:
        if a.configs:
            configs = {k: ABLATIONS[k] for k in a.configs}
        elif a.full_ablation:
            configs = ABLATIONS
        elif a.ablation:
            configs = {k: ABLATIONS[k] for k in SUBSET}
        else:
            configs = {'baseline': {}}
        todo = {k: v for k, v in configs.items() if (name, k) not in done}
        if not todo:
            continue
        loader = Dataset_Loader(name, '', name=name)
        loader.dataset_source_folder_path = a.data_dir
        loader.dataset_source_file_name = FILES[name]
        data = loader.load()
        for tag, cfg in todo.items():
            cfg = dict(cfg)
            if a.quick:
                cfg['epochs'] = 3
            results.append(run_one(name, data, cfg, tag, a.out_dir))
            save_results()   # saved after every finished run