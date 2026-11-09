'''
CNN for CIFAR-10: input 3x32x32 (RGB), 10 classes (colored objects).
'''
from Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    def __init__(self, mName='CNN-CIFAR', mDescription='CNN for colored objects', **cfg):
        base = dict(conv_channels=[32, 64, 128], kernel_size=3, padding=1, stride=1,
                    pool='max', hidden_dims=[256], dropout=0.3, epochs=20, batch_size=64)
        super().__init__(mName, mDescription, (3, 32, 32), 10, **{**base, **cfg})