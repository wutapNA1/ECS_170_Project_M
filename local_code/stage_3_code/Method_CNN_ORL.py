'''
CNN for ORL faces: input 1x112x92 (one channel), 40 classes (persons).
Only 360 training images, so stronger dropout and more epochs are used.
'''
from Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    def __init__(self, mName='CNN-ORL', mDescription='CNN for face recognition', **cfg):
        base = dict(conv_channels=[16, 32, 64], kernel_size=3, padding=1, stride=1,
                    pool='max', hidden_dims=[256], dropout=0.5, epochs=40, batch_size=32)
        super().__init__(mName, mDescription, (1, 112, 92), 40, **{**base, **cfg})