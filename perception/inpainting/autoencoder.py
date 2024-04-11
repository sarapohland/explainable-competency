import random
import numpy as np

import torch
import torch.nn as nn

from perception.network.model import NeuralNet

torch.manual_seed(2)
torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class AutoEncoder(nn.Module):
    def __init__(self, encoder_args, decoder_args):
        super().__init__()

        self.encoder = NeuralNet(encoder_args)
        self.decoder = NeuralNet(decoder_args)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y