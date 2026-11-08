'''
Concrete MethodModule class for a specific learning MethodModule (Stage 2: PyTorch MLP)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Method_MLP(method, nn.Module):
    """
    A configurable multi-layer perceptron implemented with PyTorch.

    Compared with the Stage 1 MLP (a toy model with a 4 -> 4 -> 2 architecture), this
    version is adapted to the Stage 2 data set and extended for task 2-4:

      * the input / output size follows the data (784 -> 10 instead of 4 -> 2);
      * `hidden_sizes` is a list, so a deeper model only takes a different list
        (e.g. [512, 256, 128] builds three hidden layers);
      * BatchNorm / Dropout / activation function are switches;
      * the network returns *logits* and the softmax is applied only when probabilities
        are required, so CrossEntropyLoss, NLLLoss and BCEWithLogitsLoss can all be used
        without the "double softmax" of the Stage 1 code;
      * mini-batch training, weight decay, learning rate schedule and early stopping are
        supported;
      * the training / testing loss and accuracy of every epoch are recorded in
        `self.history`, which is what the learning convergence curve plots are drawn from.
    """

    data = None

    # ---- training settings ---------------------------------------------------------
    max_epoch = 8
    learning_rate = 1e-3
    batch_size = 128
    weight_decay = 0.0
    early_stop_patience = 0        # 0 disables early stopping
    seed = 1
    verbose = True

    # ---- device ---------------------------------------------------------------------
    # `device` is a property: None means "use the GPU when it is available, otherwise the
    # CPU", and assigning 'cpu' / 'cuda' at any time moves the network to that device, e.g.
    #     method_obj.device = 'cpu'
    # The scripts of Stage 2 expose it through the STAGE2_DEVICE environment variable.

    # ---- model architecture -------------------------------------------------------
    n_feature = 784
    n_class = 10
    hidden_sizes = [128]
    activation_name = 'relu'
    dropout = 0.0
    batch_norm = False

    # ---- loss / optimizer ---------------------------------------------------------
    loss_name = 'cross_entropy'    # 'cross_entropy' | 'nll' | 'bce'
    optimizer_name = 'adam'        # 'adam' | 'adamw' | 'sgd' | 'rmsprop' | 'adagrad'
    momentum = 0.9
    lr_scheduler_name = None       # None | 'step'
    lr_scheduler_step = 5
    lr_scheduler_gamma = 0.5

    # ---- recorded training process -------------------------------------------------
    history = None
    best_epoch = None

    def __init__(self, mName=None, mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.network = self._build_network()
        self.loss_function = self._build_loss_function()
        self.history = {'epoch': [], 'train_loss': [], 'train_accuracy': [],
                        'test_loss': [], 'test_accuracy': []}
        self.best_epoch = None
        # put the parameters of the network on the GPU (when one is available)
        self.device_request = None
        self._device = torch.device('cpu')
        self.device = None

    # ------------------------------------------------------------------ model building
    def _activation(self):
        name = str(self.activation_name).lower()
        if name == 'relu':
            return nn.ReLU()
        if name == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        if name == 'tanh':
            return nn.Tanh()
        if name == 'sigmoid':
            return nn.Sigmoid()
        raise ValueError('unsupported activation function: ' + str(self.activation_name))

    def _build_network(self):
        '''Build the MLP: input -> [Linear (+BN) (activation) (dropout)] * n -> output logits'''
        layers = []
        in_size = self.n_feature
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_size, int(hidden_size)))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(int(hidden_size)))
            layers.append(self._activation())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            in_size = int(hidden_size)
        layers.append(nn.Linear(in_size, self.n_class))
        return nn.Sequential(*layers)

    def _build_loss_function(self):
        name = str(self.loss_name).lower()
        if name in ('cross_entropy', 'ce'):
            # check here for the nn.CrossEntropyLoss doc:
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            return nn.CrossEntropyLoss()
        if name in ('nll', 'nll_loss'):
            # NLLLoss expects log-probabilities, see _compute_loss below
            return nn.NLLLoss()
        if name in ('bce', 'bce_with_logits'):
            return nn.BCEWithLogitsLoss()
        raise ValueError('unsupported loss function: ' + str(self.loss_name))

    def _build_optimizer(self):
        name = str(self.optimizer_name).lower()
        parameters = self.parameters()
        if name == 'adam':
            return torch.optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        if name == 'adamw':
            return torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        if name == 'sgd':
            return torch.optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        if name == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=self.learning_rate, momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        if name == 'adagrad':
            return torch.optim.Adagrad(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        raise ValueError('unsupported optimizer: ' + str(self.optimizer_name))

    def _build_scheduler(self, optimizer):
        if self.lr_scheduler_name is None:
            return None
        name = str(self.lr_scheduler_name).lower()
        if name == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_step,
                                                   gamma=self.lr_scheduler_gamma)
        raise ValueError('unsupported learning rate scheduler: ' + str(self.lr_scheduler_name))

    # ------------------------------------------------------------------ device
    def _resolve_device(self):
        '''Pick the GPU of the ROCm / CUDA build of PyTorch when one is available'''
        requested = self.device_request
        if isinstance(requested, str) and requested:
            return torch.device(requested)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.device_request = value
        self._device = self._resolve_device()
        # move the parameters of the network to the requested device
        self.to(self._device)

    def device_summary(self):
        '''A readable description of the device used for training'''
        name = ''
        if self.device.type == 'cuda':
            try:
                name = torch.cuda.get_device_name(self.device)
            except Exception:
                name = ''
        return str(self.device) + (' (' + name + ')' if name else '')

    def _to_device(self, array, dtype):
        '''Build a torch tensor of `array` directly on the training device'''
        return torch.as_tensor(np.asarray(array), dtype=dtype).to(self.device)

    # ------------------------------------------------------------------ forward / loss
    def forward(self, x):
        '''Forward propagation, the output layer produces raw logits'''
        # self.network(x) will be a n x 10 tensor:
        # n (the number of input instances): 0th dimension; 10 (the number of classes): 1st dimension
        return self.network(x)

    def _compute_loss(self, output, target):
        name = str(self.loss_name).lower()
        if name in ('nll', 'nll_loss'):
            # NLLLoss needs log-probabilities, so the softmax is applied here instead of
            # applying it in forward() and then taking the log again
            return self.loss_function(torch.log_softmax(output, dim=1), target)
        if name in ('bce', 'bce_with_logits'):
            target_one_hot = torch.zeros(target.shape[0], self.n_class, device=target.device)
            target_one_hot.scatter_(1, target.view(-1, 1), 1.0)
            return self.loss_function(output, target_one_hot)
        return self.loss_function(output, target)

    # ------------------------------------------------------------------ training
    def _evaluate_split(self, X, y):
        '''Compute the loss and the accuracy of one split without updating the parameters'''
        self.eval()
        X_tensor = self._to_device(X, torch.float32)
        y_tensor = self._to_device(y, torch.int64)
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], max(self.batch_size, 512)):
                batch_X = X_tensor[start:start + max(self.batch_size, 512)]
                batch_y = y_tensor[start:start + max(self.batch_size, 512)]
                output = self.forward(batch_X)
                total_loss += self._compute_loss(output, batch_y).item() * batch_X.shape[0]
                correct += (output.max(1)[1] == batch_y).sum().item()
        return total_loss / X_tensor.shape[0], correct / X_tensor.shape[0]

    def train_model(self, X, y, X_test=None, y_test=None):
        '''
        Mini-batch training.

        `self.history` records the training / testing loss and accuracy after every epoch,
        these are the numbers plotted by the learning convergence curves.
        '''
        # NOTE: method.train() is overridden by this module, so the torch mode switch must
        # be called through nn.Module explicitly
        self.loss_function = self._build_loss_function()
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        X_tensor = self._to_device(X, torch.float32)
        y_tensor = self._to_device(y, torch.int64)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # a remainder of exactly one instance would break BatchNorm1d in training mode
        drop_last = bool(self.batch_norm) and (X_tensor.shape[0] % self.batch_size == 1)
        # the tensors already live on the training device, so there is nothing to pin
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size,
                            shuffle=True, generator=generator, drop_last=drop_last)

        best_test_loss = float('inf')
        best_state = None
        patience_count = 0

        for epoch in range(1, self.max_epoch + 1):
            nn.Module.train(self)
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_X, batch_y in loader:
                output = self.forward(batch_X)
                loss = self._compute_loss(output, batch_y)

                # check here for the gradient init doc:
                # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_X.shape[0]
                correct += (output.max(1)[1] == batch_y).sum().item()
                total += batch_X.shape[0]

            train_loss = total_loss / total
            train_accuracy = correct / total
            test_loss, test_accuracy = float('nan'), float('nan')
            if X_test is not None and len(X_test) > 0:
                test_loss, test_accuracy = self._evaluate_split(X_test, y_test)

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_accuracy)

            if self.verbose:
                print('Epoch: ' + str(epoch) + '/' + str(self.max_epoch)
                      + ', train loss: ' + str(round(train_loss, 4))
                      + ', train accuracy: ' + str(round(train_accuracy, 4))
                      + ', test loss: ' + str(round(test_loss, 4))
                      + ', test accuracy: ' + str(round(test_accuracy, 4)))

            if scheduler is not None:
                scheduler.step()

            if self.early_stop_patience and self.early_stop_patience > 0:
                if test_loss < best_test_loss - 1e-5:
                    best_test_loss = test_loss
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    self.best_epoch = epoch
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.early_stop_patience:
                        print('early stopping at epoch ' + str(epoch) + ', restoring epoch '
                              + str(self.best_epoch))
                        if best_state is not None:
                            self.load_state_dict(best_state)
                        break

        if self.early_stop_patience and self.early_stop_patience > 0 and best_state is None:
            self.best_epoch = int(np.argmin(self.history['test_loss'])) + 1

    # The Stage 1 template calls the training entry point `train`, while torch's nn.Module
    # also has a `train(mode)` function that switches between training and evaluation mode
    # (nn.Module.eval() calls self.train(False)).  This method accepts both call styles so
    # that neither the template naming nor `self.eval()` breaks.
    def train(self, X=None, y=None):
        if isinstance(X, bool):
            return nn.Module.train(self, X)
        if X is None:
            raise TypeError('train() expects the training data, use train_model() instead')
        return self.train_model(X, y)

    def test(self, X):
        # do the testing, and return the predicted labels
        self.eval()
        with torch.no_grad():
            output = self.forward(self._to_device(X, torch.float32))
            # convert the output of the network to the corresponding labels
            # instances will get the labels corresponding to the largest score
            return output.max(1)[1].cpu()

    def predict_proba(self, X):
        '''Return the class probability distribution of every instance'''
        self.eval()
        with torch.no_grad():
            output = self.forward(self._to_device(X, torch.float32))
            if str(self.loss_name).lower() in ('bce', 'bce_with_logits'):
                return torch.sigmoid(output).cpu()
            return torch.softmax(output, dim=1).cpu()

    def config_summary(self):
        return {'hidden_sizes': list(self.hidden_sizes),
                'activation': self.activation_name,
                'dropout': self.dropout,
                'batch_norm': self.batch_norm,
                'loss': self.loss_name,
                'optimizer': self.optimizer_name,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'max_epoch': self.max_epoch,
                'lr_scheduler': self.lr_scheduler_name,
                'device': self.device_summary()}

    def run(self):
        print('method running...')
        print('training device: ' + self.device_summary() + ', torch ' + torch.__version__)
        print('--start training...')
        start_time = time.time()
        self.train_model(self.data['train']['X'], self.data['train']['y'],
                         self.data['test']['X'], self.data['test']['y'])
        self.method_training_time = time.time() - start_time
        print('--start testing...')
        test_start_time = time.time()
        pred_y = self.test(self.data['test']['X'])
        pred_proba = self.predict_proba(self.data['test']['X'])
        self.method_testing_time = time.time() - test_start_time
        print('training time: ' + str(round(self.method_training_time, 2)) + 's')
        return {'pred_y': pred_y.numpy(),
                'true_y': np.asarray(self.data['test']['y']),
                'pred_proba': pred_proba.numpy(),
                'history': self.history,
                'config': self.config_summary(),
                'training_time': self.method_training_time}
