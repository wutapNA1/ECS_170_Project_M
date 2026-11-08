'''
Stage 2 - tasks 2-3 and 2-4: train an MLP with PyTorch on the Stage 2 data set,
apply the learned model to the testing set, draw the learning convergence curves,
report the multi-class evaluation metrics, and finally compare several model
architectures / loss functions / optimizers to increase the performance score.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import sys

# make the `local_code` package importable no matter which folder this script is started from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Learning_Curve import save_learning_curve

# ---- Stage 2 data set ---------------------------------------------------------------
# path of the downloaded Stage 2 data (it holds train.csv and test.csv); change the default
# below, or set the STAGE2_DATA_FOLDER environment variable, to point at your own copy
DEFAULT_STAGE2_DATA_FOLDER = r'D:\dl\Compressed\stage_2_data\stage_2_data'
STAGE2_DATA_FOLDER = os.environ.get('STAGE2_DATA_FOLDER', DEFAULT_STAGE2_DATA_FOLDER)
# results and plots of Stage 2
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')
PLOT_FOLDER = os.path.join(RESULT_FOLDER, 'plots')

# the training interpreter of this project is
#   C:\Users\zmzbw\AppData\Local\Programs\Python\Python312\python.exe
# which ships torch 2.13.0+rocm10.0.0, so the MLP is trained on the GPU through ROCm
# (run the script with that interpreter, e.g.
#     "C:\Users\zmzbw\AppData\Local\Programs\Python\Python312\python.exe" script/stage_2_script/script_mlp.py
# ). Set STAGE2_DEVICE=cpu to force a CPU run.
TRAINING_INTERPRETER = r'C:\Users\zmzbw\AppData\Local\Programs\Python\Python312\python.exe'

# optional quick-run switches: 0 keeps the full data set / the configured number of epochs
TRAIN_LIMIT = int(os.environ.get('STAGE2_TRAIN_LIMIT', '0'))
TEST_LIMIT = int(os.environ.get('STAGE2_TEST_LIMIT', '0'))
EPOCH_OVERRIDE = int(os.environ.get('STAGE2_EPOCHS', '0'))
# 'cpu' / 'cuda', empty means "GPU when available, otherwise CPU"
DEVICE_OVERRIDE = os.environ.get('STAGE2_DEVICE', '')

# ---- task 2-4: the experiments -------------------------------------------------------
# the first model is the simple baseline (one hidden layer, CrossEntropyLoss, Adam), the
# following ones change the architecture (more layers / wider layers / BatchNorm / Dropout),
# the loss function (NLLLoss) and the optimizer (SGD with momentum) to increase the score
EXPERIMENTS = [
    {'name': 'MLP_baseline',
     'description': '1 hidden layer (128)',
     'hidden_sizes': [128],
     'dropout': 0.0,
     'batch_norm': False,
     'loss_name': 'cross_entropy',
     'optimizer_name': 'adam',
     'learning_rate': 1e-3,
     'weight_decay': 0.0,
     'batch_size': 128,
     'max_epoch': 10,
     'lr_scheduler_name': None},
    {'name': 'MLP_deeper',
     'description': '2 hidden layers (512, 256) + BatchNorm + Dropout, Adam, lr 2e-3',
     'hidden_sizes': [512, 256],
     'dropout': 0.1,
     'batch_norm': True,
     'loss_name': 'cross_entropy',
     'optimizer_name': 'adam',
     'learning_rate': 2e-3,
     'weight_decay': 1e-4,
     'batch_size': 256,
     'max_epoch': 20,
     'lr_scheduler_name': 'step'},
    {'name': 'MLP_three_layers_sgd_nll',
     'description': '3 hidden layers (512, 256, 128) + BatchNorm + Dropout, NLLLoss, SGD momentum 0.9, lr 0.1',
     'hidden_sizes': [512, 256, 128],
     'dropout': 0.1,
     'batch_norm': True,
     'loss_name': 'nll',
     'optimizer_name': 'sgd',
     'momentum': 0.9,
     'learning_rate': 0.1,
     'weight_decay': 1e-4,
     'batch_size': 256,
     'max_epoch': 25,
     'lr_scheduler_name': 'step'},
]


def run_one_experiment(config, seed=1):
    '''Train and test one MLP configuration, then save its result and its convergence curves'''
    print('')
    print('************ Start: ' + config['name'] + ' ************')
    print('architecture: ' + config['description'])

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- objection initialization section ------------------------------
    data_obj = Dataset_Loader('stage 2 data set', 'mnist style train/test csv files')
    data_obj.dataset_source_folder_path = STAGE2_DATA_FOLDER
    data_obj.dataset_source_file_name = 'train.csv'
    data_obj.dataset_test_file_name = 'test.csv'
    data_obj.max_train_samples = TRAIN_LIMIT
    data_obj.max_test_samples = TEST_LIMIT

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.hidden_sizes = config['hidden_sizes']
    method_obj.dropout = config['dropout']
    method_obj.batch_norm = config['batch_norm']
    method_obj.loss_name = config['loss_name']
    method_obj.optimizer_name = config['optimizer_name']
    method_obj.momentum = config.get('momentum', 0.9)
    method_obj.learning_rate = config['learning_rate']
    method_obj.weight_decay = config['weight_decay']
    method_obj.batch_size = config['batch_size']
    method_obj.max_epoch = EPOCH_OVERRIDE if EPOCH_OVERRIDE > 0 else config['max_epoch']
    method_obj.lr_scheduler_name = config['lr_scheduler_name']
    method_obj.seed = seed
    method_obj.device = DEVICE_OVERRIDE or None

    # the Stage 2 data set is pre-partitioned, so the result of one run is stored with
    # fold_count = None, i.e. in `result/stage_2_result/<experiment name>_None`
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER
    result_obj.result_destination_file_name = config['name']

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')

    evaluate_obj = Evaluate_Metrics('multi-class metrics', 'accuracy + weighted/macro/micro P/R/F1')
    # ---------------------------------------------------------------------

    # ---- running section -------------------------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()
    evaluate_obj.print_metrics(metrics, 'testing set performance of ' + config['name'])
    evaluate_obj.print_report('per class performance of ' + config['name'])

    # task 2-3: the learning convergence curves of this run
    plot_paths = save_learning_curve(method_obj.history, PLOT_FOLDER,
                                     file_stem=config['name'],
                                     title=config['name'] + ': ' + config['description'])
    print('************ Finish: ' + config['name'] + ' ************')
    # ---------------------------------------------------------------------

    return {'name': config['name'],
            'description': config['description'],
            'metrics': metrics,
            'plot_paths': plot_paths,
            'config': method_obj.config_summary()}


def main():
    print('python interpreter: ' + sys.executable)
    print('torch: ' + torch.__version__ + ', GPU available: ' + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('GPU: ' + torch.cuda.get_device_name(0))
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(TRAINING_INTERPRETER):
        print('note: the project interpreter is ' + TRAINING_INTERPRETER
              + ', this run uses ' + sys.executable)

    summaries = []
    for config in EXPERIMENTS:
        summaries.append(run_one_experiment(config))

    # ---- task 2-4: comparison of the experiments -------------------------
    print('')
    print('************ Overall Performance ************')
    print('%-30s %-10s %-10s %-10s %-10s' % ('experiment', 'accuracy', 'precision_w', 'recall_w', 'f1_w'))
    for summary in summaries:
        metrics = summary['metrics']
        print('%-30s %-10.4f %-10.4f %-10.4f %-10.4f' % (summary['name'],
                                                         metrics['accuracy'],
                                                         metrics['precision_weighted'],
                                                         metrics['recall_weighted'],
                                                         metrics['f1_weighted']))
    best_summary = max(summaries, key=lambda item: item['metrics']['accuracy'])
    print('best experiment: ' + best_summary['name'] + ' with accuracy '
          + str(round(best_summary['metrics']['accuracy'], 4)))
    print('************ Finish ************')


if __name__ == '__main__':
    main()
