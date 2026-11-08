'''
Concrete Evaluate class for the multi-class evaluation metrics of Stage 2
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class Evaluate_Metrics(evaluate):
    """
    Evaluation module for the multi-class classification task of Stage 2.

    Stage 2 is a *multi-class* classification task, hence the binary Precision / Recall /
    F1 scores are not meaningful and the "weighted", "macro" and "micro" averaged versions
    of these metrics are computed instead:

      * accuracy            : ratio of correctly predicted instances
      * weighted average    : average weighted by the number of instances of each class,
                              i.e. it accounts for class imbalance
      * macro average       : unweighted average over the classes, every class counts the same
      * micro average       : global average over all instances (equals accuracy for the
                              single-label multi-class case, kept for completeness)

    The confusion matrix is returned as well, so that the per-class behaviour can be
    inspected in the report.
    """

    data = None
    n_class = 10

    def evaluate(self):
        print('evaluating performance...')
        true_y = np.asarray(self.data['true_y']).astype(np.int64).ravel()
        pred_y = np.asarray(self.data['pred_y']).astype(np.int64).ravel()
        labels = list(range(self.n_class))

        metrics = {
            'accuracy': float(accuracy_score(true_y, pred_y)),
            'precision_weighted': float(precision_score(true_y, pred_y, labels=labels, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(true_y, pred_y, labels=labels, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(true_y, pred_y, labels=labels, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(true_y, pred_y, labels=labels, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(true_y, pred_y, labels=labels, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(true_y, pred_y, labels=labels, average='macro', zero_division=0)),
            'precision_micro': float(precision_score(true_y, pred_y, labels=labels, average='micro', zero_division=0)),
            'recall_micro': float(recall_score(true_y, pred_y, labels=labels, average='micro', zero_division=0)),
            'f1_micro': float(f1_score(true_y, pred_y, labels=labels, average='micro', zero_division=0)),
            'confusion_matrix': confusion_matrix(true_y, pred_y, labels=labels),
        }
        return metrics

    def print_metrics(self, metrics, title='evaluation result'):
        '''Print the metric dictionary returned by evaluate() in a readable way'''
        print('************ ' + str(title) + ' ************')
        print('accuracy            : ' + str(round(metrics['accuracy'], 4)))
        print('precision (weighted): ' + str(round(metrics['precision_weighted'], 4)))
        print('recall    (weighted): ' + str(round(metrics['recall_weighted'], 4)))
        print('f1        (weighted): ' + str(round(metrics['f1_weighted'], 4)))
        print('precision (macro)   : ' + str(round(metrics['precision_macro'], 4)))
        print('recall    (macro)   : ' + str(round(metrics['recall_macro'], 4)))
        print('f1        (macro)   : ' + str(round(metrics['f1_macro'], 4)))
        print('precision (micro)   : ' + str(round(metrics['precision_micro'], 4)))
        print('recall    (micro)   : ' + str(round(metrics['recall_micro'], 4)))
        print('f1        (micro)   : ' + str(round(metrics['f1_micro'], 4)))

    def print_report(self, title='per class report'):
        '''Print the per class precision / recall / f1 of the current data'''
        print('************ ' + str(title) + ' ************')
        true_y = np.asarray(self.data['true_y']).astype(np.int64).ravel()
        pred_y = np.asarray(self.data['pred_y']).astype(np.int64).ravel()
        print(classification_report(true_y, pred_y, labels=list(range(self.n_class)), zero_division=0))
