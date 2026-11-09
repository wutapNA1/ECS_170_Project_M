from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from code.base_class.evaluate import evaluate
except ImportError:
    class evaluate:
        def __init__(self, eName=None, eDescription=None):
            self.evaluate_name, self.evaluate_description, self.data = eName, eDescription, None


class Evaluate_Metrics(evaluate):
    def evaluate(self):
        t, p = self.data['true_y'], self.data['pred_y']
        return {'accuracy': accuracy_score(t, p),
                'precision': precision_score(t, p, average='weighted', zero_division=0),
                'recall': recall_score(t, p, average='weighted', zero_division=0),
                'f1': f1_score(t, p, average='weighted', zero_division=0)}