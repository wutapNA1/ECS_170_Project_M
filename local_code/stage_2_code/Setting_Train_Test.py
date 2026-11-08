'''
Concrete SettingModule class for a specific experimental SettingModule (Stage 2)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test(setting):
    """
    Stage 2 experiment setting.

    The Stage 2 data set is already partitioned into a training set and a testing set
    (train.csv / test.csv), therefore this setting only trains on the given training set
    and evaluates on the given testing set: no train/test split and no cross validation
    are performed, which replaces Setting_KFold_CV / Setting_Train_Test_Split of Stage 1.
    """

    def load_run_save_evaluate(self):
        # load the pre-partitioned training and testing sets
        loaded_data = self.dataset.load()
        print('training instances: ' + str(len(loaded_data['train']['y']))
              + ', testing instances: ' + str(len(loaded_data['test']['y'])))

        # run MethodModule on the pre-partitioned sets
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = None
        self.result.save()

        self.evaluate.data = learned_result

        # return the multi-class metric dictionary of the testing set
        return self.evaluate.evaluate()
