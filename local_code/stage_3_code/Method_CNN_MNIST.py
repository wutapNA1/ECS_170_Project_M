'''
CNN for MNIST: input 1x28x28, 10 classes (digits 0 to 9).
'''
from Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    def __init__(self, mName='CNN-MNIST', mDescription='CNN for hand-written digits', **cfg):
        base = dict(conv_channels=[32, 64], kernel_size=3, padding=1, stride=1,
                    pool='max', hidden_dims=[128], dropout=0.3, epochs=10, batch_size=64)
        super().__init__(mName, mDescription, (1, 28, 28), 10, **{**base, **cfg})