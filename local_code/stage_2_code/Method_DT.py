'''
Concrete MethodModule class for a specific learning MethodModule (Stage 2: decision tree)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from sklearn import tree


class Method_DT(method):
    data = None

    # hyper parameters of the decision tree
    criterion = 'gini'
    max_depth = None
    min_samples_leaf = 1
    random_state = 1

    def train(self, X, y):
        # check here for the decision tree classifier:
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        model = tree.DecisionTreeClassifier(criterion=self.criterion,
                                           max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf,
                                           random_state=self.random_state)
        # model variable learning has been built-in by the fit() function
        model.fit(X, y)
        return model

    def test(self, model, X):
        # check here for decision tree predict doc:
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        return model.predict(X)

    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(model, self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
