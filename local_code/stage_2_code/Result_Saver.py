'''
Concrete ResultModule class for a specific experiment ResultModule output (Stage 2)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import os
import pickle


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def _destination_path(self):
        folder = self.result_destination_folder_path or '.'
        file_name = str(self.result_destination_file_name) + '_' + str(self.fold_count)
        return os.path.join(folder, file_name)

    def save(self):
        print('saving results...')
        path = self._destination_path()
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print('  result saved to ' + path)
