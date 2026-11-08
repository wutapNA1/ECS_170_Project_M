'''
Concrete IO class for a specific dataset (Stage 2 data set)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import os
import time


class Dataset_Loader(dataset):
    """
    Dataset loader for the Stage 2 data set.

    The Stage 2 data set is the MNIST-style data set provided by the instructor and it
    comes with a *pre-partitioned* training file and testing file, therefore this loader
    returns both parts at once and the Setting module of Stage 2 does not perform any
    train/test split or cross validation.

    Each line of train.csv / test.csv contains 785 comma separated integers:

        label, feature_1, feature_2, ..., feature_784

    where the label is in {0, 1, ..., 9} and every feature is a pixel value in [0, 255].
    """

    # default location of the downloaded Stage 2 data (the folder holding train.csv / test.csv),
    # it can be overwritten by the script before calling load(); the STAGE2_DATA_FOLDER
    # environment variable takes precedence so that the repository works on other machines
    DEFAULT_DATA_FOLDER = r'D:\dl\Compressed\stage_2_data\stage_2_data'
    dataset_source_folder_path = os.environ.get('STAGE2_DATA_FOLDER', DEFAULT_DATA_FOLDER)
    dataset_source_file_name = 'train.csv'
    dataset_test_file_name = 'test.csv'

    n_feature = 784
    n_class = 10

    # scale the pixels from [0, 255] into [0, 1]
    normalize = True

    # 0 means "use every instance"; a positive value keeps that many instances, which is
    # useful when a method is too slow on the full data set (e.g. the SVM script)
    max_train_samples = 0
    max_test_samples = 0
    subset_seed = 1

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _resolve_path(self, file_name):
        if os.path.isabs(file_name):
            return file_name
        return os.path.join(self.dataset_source_folder_path or '', file_name)

    @staticmethod
    def _read_csv(path):
        try:
            data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        except Exception:
            # fall back to a plain python parser if numpy refuses to read the file
            with open(path, 'r') as f:
                rows = [line.split(',') for line in f if line.strip()]
            data = np.array(rows, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    def _load_single_file(self, file_name):
        path = self._resolve_path(file_name)
        if not os.path.exists(path):
            raise FileNotFoundError('Stage 2 data file not found: ' + path)
        print('loading data from ' + path + ' ...')
        start_time = time.time()
        raw = self._read_csv(path)
        X = raw[:, 1:]
        y = raw[:, 0].astype(np.int64)
        print('  ' + str(X.shape[0]) + ' instances, ' + str(X.shape[1]) + ' features, '
              + str(len(np.unique(y))) + ' classes, loaded in ' + str(round(time.time() - start_time, 2)) + 's')
        if X.shape[1] != self.n_feature:
            print('  warning: expected ' + str(self.n_feature) + ' features but got ' + str(X.shape[1]))
        if self.normalize:
            X = X / np.float32(255.0)
        return X.astype(np.float32), y

    def _subsample(self, X, y, max_samples):
        if not max_samples or max_samples >= X.shape[0]:
            return X, y
        generator = np.random.RandomState(self.subset_seed)
        index = np.sort(generator.choice(X.shape[0], size=max_samples, replace=False))
        return X[index], y[index]

    def load(self):
        X_train, y_train = self._load_single_file(self.dataset_source_file_name)
        X_test, y_test = self._load_single_file(self.dataset_test_file_name)
        X_train, y_train = self._subsample(X_train, y_train, self.max_train_samples)
        X_test, y_test = self._subsample(X_test, y_test, self.max_test_samples)
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
