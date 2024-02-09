import os
import json
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm

from perception.competency.alice import Alice
from perception.datasets.setup_dataloader import setup_loader

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--train_data', type=str, default='data/')
    parser.add_argument('--model_dir', type=str, default='model/classify/')
    args = parser.parse_args()

    # Collect training data
    train_loader = setup_loader(args.train_data, train=True)

    # Fit competency estimator from training data
    estimator = Alice(train_loader, args.model_dir)
    file = os.path.join(args.model_dir, 'alice.p')
    pickle.dump(estimator, open(file, 'wb'))

if __name__ == "__main__":
    main()
