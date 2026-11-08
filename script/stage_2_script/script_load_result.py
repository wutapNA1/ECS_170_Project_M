'''
Stage 2 - load the saved results of Stage 2 back from disk and report their metrics.

Because the Stage 2 data set is pre-partitioned, every run of Stage 2 stores exactly one
result, under `result/stage_2_result/<name>_None`. This script walks over the saved results,
prints the multi-class evaluation metrics of each of them again, and summarises the
convergence of the MLP runs from the stored per epoch history.
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

from local_code.stage_2_code.Result_Loader import Result_Loader
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics

RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')


def list_saved_results(folder):
    '''Return the names of the saved Stage 2 results (files ending with `_None`)'''
    if not os.path.exists(folder):
        return []
    suffix = '_' + str(None)
    names = []
    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith(suffix):
            names.append(file_name[:-len(suffix)])
    return names


def main():
    names = list_saved_results(RESULT_FOLDER)
    if not names:
        print('no saved Stage 2 result found in ' + RESULT_FOLDER)
        print('run script_mlp.py / script_decision_tree.py / script_svm.py first')
        return

    summary = []
    for name in names:
        print('')
        print('************ ' + name + ' ************')
        result_obj = Result_Loader('loader', '')
        result_obj.result_destination_folder_path = RESULT_FOLDER
        result_obj.result_destination_file_name = name
        result_obj.fold_count = None
        result_obj.load()
        learned_result = result_obj.data

        evaluate_obj = Evaluate_Metrics('multi-class metrics', 'accuracy + weighted/macro/micro P/R/F1')
        evaluate_obj.data = learned_result
        metrics = evaluate_obj.evaluate()
        evaluate_obj.print_metrics(metrics, 'testing set performance of ' + name)

        history = learned_result.get('history')
        if history and len(history.get('epoch', [])) > 0:
            print('training converged after ' + str(len(history['epoch'])) + ' epochs')
            print('  train loss: ' + str(round(history['train_loss'][0], 4)) + ' -> '
                  + str(round(history['train_loss'][-1], 4))
                  + ', test accuracy: ' + str(round(history['test_accuracy'][0], 4)) + ' -> '
                  + str(round(history['test_accuracy'][-1], 4)))
        if learned_result.get('config'):
            print('model configuration: ' + str(learned_result['config']))

        summary.append((name, metrics['accuracy'], metrics['precision_weighted'],
                        metrics['recall_weighted'], metrics['f1_weighted']))

    print('')
    print('************ Overall Performance ************')
    print('%-30s %-10s %-10s %-10s %-10s' % ('result', 'accuracy', 'precision_w', 'recall_w', 'f1_w'))
    for name, accuracy, precision_w, recall_w, f1_w in summary:
        print('%-30s %-10.4f %-10.4f %-10.4f %-10.4f' % (name, accuracy, precision_w, recall_w, f1_w))
    print('************ Finish ************')


if __name__ == '__main__':
    main()
