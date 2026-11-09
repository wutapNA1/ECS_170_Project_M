'''
Concrete IO class for the stage 3 image datasets (MNIST, ORL, CIFAR).
File format: a pickle holding {'train': [{'image','label'},...], 'test': [...]}.
'''
import pickle
import numpy as np

try:
    from code.base_class.dataset import dataset
except ImportError:  # fallback so the file also runs stand-alone
    class dataset:
        def __init__(self, dName=None, dDescription=None):
            self.dataset_name = dName
            self.dataset_description = dDescription
            self.dataset_source_folder_path = None
            self.dataset_source_file_name = None
            self.data = None


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, name='MNIST'):
        super().__init__(dName, dDescription)
        self.name = name.upper()  # 'MNIST' | 'ORL' | 'CIFAR'

    @staticmethod
    def _stack(split):
        X = np.stack([np.array(d['image']) for d in split]).astype(np.float32)
        y = np.array([d['label'] for d in split], dtype=np.int64)
        return X, y

    def _to_nchw(self, X):
        if X.ndim == 3:                      # N, H, W  (MNIST)
            X = X[:, None, :, :]
        elif X.ndim == 4:                    # N, H, W, C
            if self.name == 'ORL':           # R=G=B, one channel is enough
                X = X[..., :1]
            X = np.transpose(X, (0, 3, 1, 2))
        return X / 255.0

    def load(self):
        print('loading %s data...' % self.name)
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        X_train, y_train = self._stack(raw['train'])
        X_test, y_test = self._stack(raw['test'])
        X_train, X_test = self._to_nchw(X_train), self._to_nchw(X_test)

        # ORL labels are 1..40, shift so classes start at 0
        shift = y_train.min()
        y_train, y_test = y_train - shift, y_test - shift

        # per-channel standardisation using training statistics
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        X_train, X_test = (X_train - mean) / std, (X_test - mean) / std

        self.data = {'train': {'X': X_train.astype(np.float32), 'y': y_train},
                     'test': {'X': X_test.astype(np.float32), 'y': y_test}}
        print('train', X_train.shape, 'test', X_test.shape,
              'classes', len(np.unique(y_train)))
        return self.data