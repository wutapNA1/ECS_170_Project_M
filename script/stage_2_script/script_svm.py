'''
Stage 2 - support vector machine baseline on the Stage 2 data set.

The Stage 1 script of this method ran a k-fold cross validation on the toy data file.
Stage 2 comes with a pre-partitioned training set and testing set, so this script trains
the SVM on train.csv and evaluates it on test.csv, and it reports the multi-class evaluation
metrics (accuracy plus the weighted / macro / micro precision, recall and F1).
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
from local_code.stage_2_code.Method_SVM import Method_SVM
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics

# ---- Stage 2 data set ---------------------------------------------------------------
# path of the downloaded Stage 2 data (it holds train.csv and test.csv); change the default
# below, or set the STAGE2_DATA_FOLDER environment variable, to point at your own copy
DEFAULT_STAGE2_DATA_FOLDER = r'D:\dl\Compressed\stage_2_data\stage_2_data'
STAGE2_DATA_FOLDER = os.environ.get('STAGE2_DATA_FOLDER', DEFAULT_STAGE2_DATA_FOLDER)
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')

# the training cost of sklearn.svm.SVC grows quadratically (and worse) with the number of
# training instances, so many instances are used instead of the 60,000 training instances
# (set it to 0 to use the whole training set, which needs a lot of time and memory)
SVM_MAX_TRAIN_SAMPLES = 8000

# optional quick-run switches: 0 keeps the value above / the full data set
TRAIN_LIMIT = int(os.environ.get('STAGE2_TRAIN_LIMIT', '0'))
TEST_LIMIT = int(os.environ.get('STAGE2_TEST_LIMIT', '0'))


def main():
    # limit of the training instances really used by this run
    train_limit = SVM_MAX_TRAIN_SAMPLES
    if TRAIN_LIMIT > 0 and (train_limit == 0 or TRAIN_LIMIT < train_limit):
        train_limit = TRAIN_LIMIT

    # ---- parameter section --------------------------------------------
    np.random.seed(2)
    # -------------------------------------------------------------------

    # ---- objection initialization section ------------------------------
    data_obj = Dataset_Loader('stage 2 data set', 'mnist style train/test csv files')
    data_obj.dataset_source_folder_path = STAGE2_DATA_FOLDER
    data_obj.dataset_source_file_name = 'train.csv'
    data_obj.dataset_test_file_name = 'test.csv'
    data_obj.max_train_samples = train_limit
    data_obj.max_test_samples = TEST_LIMIT

    method_obj = Method_SVM('support vector machine', '')
    method_obj.c = 1.0
    method_obj.kernel = 'rbf'
    method_obj.gamma = 'scale'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER
    result_obj.result_destination_file_name = 'SVM'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')

    evaluate_obj = Evaluate_Metrics('multi-class metrics', 'accuracy + weighted/macro/micro P/R/F1')
    # --------------------------------------------------------------------

    # ---- running section -----------------------------------------------
    print('************ Start ************')
    print('SVM is trained on ' + str(train_limit if train_limit > 0 else 'all')
          + ' training instances (SVC costs grow with the number of instances)')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()
    evaluate_obj.print_metrics(metrics, 'testing set performance of the SVM')
    evaluate_obj.print_report('per class performance of the SVM')
    print('************ Overall Performance ************')
    print('SVM Accuracy: ' + str(round(metrics['accuracy'], 4))
          + ', weighted F1: ' + str(round(metrics['f1_weighted'], 4))
          + ', macro F1: ' + str(round(metrics['f1_macro'], 4)))
    print('************ Finish ************')
    # --------------------------------------------------------------------


if __name__ == '__main__':
    main()
