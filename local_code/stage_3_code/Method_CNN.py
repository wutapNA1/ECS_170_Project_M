'''
Shared CNN implementation (inherits method from the base class and nn.Module).
Dataset specific classes live in Method_CNN_MNIST.py, Method_CNN_ORL.py, Method_CNN_CIFAR.py.
'''
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from code.base_class.method import method
except ImportError:
    class method:
        def __init__(self, mName=None, mDescription=None):
            self.method_name = mName
            self.method_description = mDescription
            self.data = None

DEFAULT_CFG = dict(
    conv_channels=[32, 64],   # one entry per conv block (model depth)
    kernel_size=3,
    padding=1,
    stride=1,
    pool='max',               # 'max' | 'avg' | None
    pool_size=2,
    batch_norm=True,
    hidden_dims=[128],        # fully connected hidden layers
    dropout=0.3,
    loss='ce',                # 'ce' | 'mse' | 'smooth'
    lr=1e-3,
    weight_decay=1e-4,
    epochs=15,
    batch_size=64,
    seed=1,
)


class Method_CNN(method, nn.Module):
    def __init__(self, mName, mDescription, in_shape, n_classes, **cfg):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.cfg = {**DEFAULT_CFG, **cfg}
        c = self.cfg
        torch.manual_seed(c['seed'])
        np.random.seed(c['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = n_classes

        layers, ch, h, w = [], in_shape[0], in_shape[1], in_shape[2]
        for out_ch in c['conv_channels']:
            layers.append(nn.Conv2d(ch, out_ch, c['kernel_size'], c['stride'], c['padding']))
            if c['batch_norm']:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            ch = out_ch
            if c['pool'] and min(self._out_hw(h, w)) >= c['pool_size']:
                Pool = nn.MaxPool2d if c['pool'] == 'max' else nn.AvgPool2d
                layers.append(Pool(c['pool_size']))
            h, w = self._after_block(h, w, c)
        self.features = nn.Sequential(*layers)

        self.features.eval()  # eval mode: BatchNorm rejects a batch of 1 when the map is 1x1
        with torch.no_grad():  # infer flattened size
            flat = self.features(torch.zeros(1, *in_shape)).numel()
        self.features.train()
        fc, d = [nn.Flatten()], flat
        for hd in c['hidden_dims']:
            fc += [nn.Linear(d, hd), nn.ReLU(), nn.Dropout(c['dropout'])]
            d = hd
        fc.append(nn.Linear(d, n_classes))
        self.classifier = nn.Sequential(*fc)
        self.to(self.device)
        self.history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    # helpers to track spatial size: H_out = floor((H + 2P - K)/S) + 1
    def _out_hw(self, h, w):
        c = self.cfg
        f = lambda n: (n + 2 * c['padding'] - c['kernel_size']) // c['stride'] + 1
        return f(h), f(w)

    def _after_block(self, h, w, c):
        h, w = self._out_hw(h, w)
        if c['pool'] and min(h, w) >= c['pool_size']:
            h, w = h // c['pool_size'], w // c['pool_size']
        return h, w

    def forward(self, x):
        return self.classifier(self.features(x))

    def _loss(self, out, y):
        t = self.cfg['loss']
        if t == 'mse':
            return nn.functional.mse_loss(torch.softmax(out, 1),
                                          nn.functional.one_hot(y, self.n_classes).float())
        if t == 'smooth':
            return nn.functional.cross_entropy(out, y, label_smoothing=0.1)
        return nn.functional.cross_entropy(out, y)

    def _loader(self, X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=self.cfg['batch_size'], shuffle=shuffle)

    @torch.no_grad()
    def _evaluate(self, loader):
        self.eval()
        loss, correct, n, preds = 0.0, 0, 0, []
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            out = self(xb)
            loss += self._loss(out, yb).item() * len(yb)
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            n += len(yb)
            preds.append(p.cpu())
        return loss / n, correct / n, torch.cat(preds).numpy()

    def train_model(self, X, y, X_test, y_test):
        c = self.cfg
        opt = torch.optim.Adam(self.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        tr, tr_eval, te = self._loader(X, y, True), self._loader(X, y, False), self._loader(X_test, y_test, False)
        for ep in range(c['epochs']):
            self.train()
            for xb, yb in tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                self._loss(self(xb), yb).backward()
                opt.step()
            trl, tra, _ = self._evaluate(tr_eval)
            tel, tea, _ = self._evaluate(te)
            for k, v in zip(self.history, (trl, tel, tra, tea)):
                self.history[k].append(v)
            print('epoch %2d | train loss %.4f acc %.4f | test loss %.4f acc %.4f'
                  % (ep + 1, trl, tra, tel, tea))

    def test_model(self, X_test, y_test):
        return self._evaluate(self._loader(X_test, y_test, False))[2]

    def run(self):
        d = self.data
        self.train_model(d['train']['X'], d['train']['y'], d['test']['X'], d['test']['y'])
        pred = self.test_model(d['test']['X'], d['test']['y'])
        return {'pred_y': pred, 'true_y': d['test']['y']}