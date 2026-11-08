'''
Concrete ResultModule class for a specific experiment ResultModule output (Stage 2)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import os
import pickle


class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def _destination_path(self):
        folder = self.result_destination_folder_path or '.'
        file_name = str(self.result_destination_file_name) + '_' + str(self.fold_count)
        return os.path.join(folder, file_name)

    def load(self):
        print('loading results...')
        path = self._destination_path()
        if not os.path.exists(path):
            raise FileNotFoundError('result file not found: ' + path)
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print('  result loaded from ' + path)
        return self.data
