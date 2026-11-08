'''
Stage 2 - decision tree baseline on the Stage 2 data set.

The Stage 1 script of this method ran a k-fold cross validation on the toy data file.
Stage 2 comes with a pre-partitioned training set and testing set, so this script trains
the decision tree on train.csv and evaluates it on test.csv, and it reports the multi-class
evaluation metrics (accuracy plus the weighted / macro / micro precision, recall and F1).
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

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_DT import Method_DT
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics

# ---- Stage 2 data set ---------------------------------------------------------------
# path of the downloaded Stage 2 data (it holds train.csv and test.csv); change the default
# below, or set the STAGE2_DATA_FOLDER environment variable, to point at your own copy
DEFAULT_STAGE2_DATA_FOLDER = r'D:\dl\Compressed\stage_2_data\stage_2_data'
STAGE2_DATA_FOLDER = os.environ.get('STAGE2_DATA_FOLDER', DEFAULT_STAGE2_DATA_FOLDER)
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')

# optional quick-run switches: 0 keeps the full data set
TRAIN_LIMIT = int(os.environ.get('STAGE2_TRAIN_LIMIT', '0'))
TEST_LIMIT = int(os.environ.get('STAGE2_TEST_LIMIT', '0'))


def main():
    # ---- parameter section --------------------------------------------
    np.random.seed(1)
    # -------------------------------------------------------------------

    # ---- objection initialization section ------------------------------
    data_obj = Dataset_Loader('stage 2 data set', 'mnist style train/test csv files')
    data_obj.dataset_source_folder_path = STAGE2_DATA_FOLDER
    data_obj.dataset_source_file_name = 'train.csv'
    data_obj.dataset_test_file_name = 'test.csv'
    data_obj.max_train_samples = TRAIN_LIMIT
    data_obj.max_test_samples = TEST_LIMIT

    method_obj = Method_DT('decision tree', '')
    method_obj.criterion = 'gini'
    method_obj.max_depth = None
    method_obj.min_samples_leaf = 1
    method_obj.random_state = 1

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER
    result_obj.result_destination_file_name = 'DT'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')

    evaluate_obj = Evaluate_Metrics('multi-class metrics', 'accuracy + weighted/macro/micro P/R/F1')
    # --------------------------------------------------------------------

    # ---- running section -----------------------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()
    evaluate_obj.print_metrics(metrics, 'testing set performance of the decision tree')
    evaluate_obj.print_report('per class performance of the decision tree')
    print('************ Overall Performance ************')
    print('Decision Tree Accuracy: ' + str(round(metrics['accuracy'], 4))
          + ', weighted F1: ' + str(round(metrics['f1_weighted'], 4))
          + ', macro F1: ' + str(round(metrics['f1_macro'], 4)))
    print('************ Finish ************')
    # --------------------------------------------------------------------


if __name__ == '__main__':
    main()
