import random
import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(2)
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class NeuralNet(nn.Module):
    def __init__(self, layer_args):
        super().__init__()

        layers = []
        for args in layer_args:
            n_in = args['n_in'] if 'n_in' in args else None
            n_out = args['n_out'] if 'n_out' in args else None
            kernel = args['kernel'] if 'kernel' in args else None
            stride = args['stride'] if 'stride' in args else 1
            pad = args['pad'] if 'pad' in args else 0
            out_pad = args['out_pad'] if 'out_pad' in args else 0
            mode = args['mode'] if 'mode' in args else None
            prob = args['prob'] if 'prob' in args else None
            channels = args['channels'] if 'channels' in args else None
            height = args['height'] if 'height' in args else None
            width = args['width'] if 'width' in args else None
            size = args['size'] if 'size' in args else None
            scale = args['scale'] if 'scale' in args else None

            # Linear (fully connected) layer
            if args['name'] == "fc":
                if n_in is None or n_out is None:
                    raise ValueError('The number of input and output features must be specified for a linear layer.')
                layers.append(nn.Linear(n_in, n_out))

            # Convolutional layer
            elif args['name'] == "conv":
                if n_in is None or n_out is None:
                    raise ValueError('The number of input and output channels must be specified for a convolutional layer.')
                if kernel is None:
                    raise ValueError('The kernel size must be specified for a convolutional layer.')
                layers.append(nn.Conv2d(n_in, n_out, kernel, stride=stride, padding=pad))

            # Transposed convolutional layer
            elif args['name'] == "transpose":
                if n_in is None or n_out is None:
                    raise ValueError('The number of input and output channels must be specified for a convolutional layer.')
                if kernel is None:
                    raise ValueError('The kernel size must be specified for a convolutional layer.')
                layers.append(nn.ConvTranspose2d(n_in, n_out, kernel, stride=stride, padding=pad, output_padding=out_pad))

            # Pooling layer
            elif args['name'] == "pool":
                if kernel is None:
                    raise ValueError('The kernel size must be specified for a pooling layer.')
                if args['mode'] == 'max':
                    layers.append(nn.MaxPool2d(kernel, stride=stride, padding=pad))
                elif args['mode'] == 'avg':
                    layers.append(nn.AvgPool2d(kernel, stride=stride, padding=pad))
                else:
                    raise NotImplementedError("Pooling mode {} is not implemented".format(args['mode']))
                
            # Upsampling layer
            elif args['name'] == "upsample":
                if size is None and scale is None:
                    raise ValueError('Either the scale factor or the target output size must be specified for an upsampling layer.')
                mode = 'nearest' if mode is None else mode
                layers.append(nn.Upsample(size=size, scale_factor=scale, mode=mode))

            # 1D batch normalization
            elif args['name'] == "norm1":
                if n_in is None:
                    raise ValueError('The number of features must be specified for 1D batch normalization.')
                layers.append(nn.BatchNorm1d(n_in))

            # 2D batch normalization
            elif args['name'] == "norm2":
                if n_in is None:
                    raise ValueError('The number of input channels must be specified for 2D batch normalization.')
                layers.append(nn.BatchNorm2d(n_in))

            elif args['name'] == "dropout":
                if prob is None:
                    raise ValueError('The probability of dropout needs to be specified.')
                layers.append(nn.Dropout(prob))

            # Flattening layer
            elif args['name'] == "flatten":
                layers.append(nn.Flatten())

            # Unflattening layer
            elif args['name'] == "unflatten":
                if height is None or width is None:
                    raise ValueError('The height and width need to be specified for reshaping.')
                if channels is None:
                    layers.append(nn.Unflatten(1, (height, width)))
                else:
                    layers.append(nn.Unflatten(1, (channels, height, width)))

            else:
                raise NotImplementedError("Layer type {} is not implemented".format(args['name']))

            if 'activation' in args:
                # ReLU activation
                if args['activation'] == 'relu':
                    layers.append(nn.ReLU())

                # Sigmoid activation
                elif args['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())

                # Hyperbolic tangent activation
                elif args['activation'] == 'tanh':
                    layers.append(nn.Tanh())

                # Softmax activation
                elif args['activation'] == "softmax":
                    layers.append(nn.Softmax(dim=1))

        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def get_feature_vector(self, x):
        y = x
        layers = self.layers[:-2]
        for layer in layers:
            y = layer(y)
        return y
