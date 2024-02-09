import os
import json
import shutil
import argparse
import numpy as np
import configparser
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from perception.network.model import NeuralNet
from perception.datasets.setup_dataloader import setup_loader

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        true = torch.ones(pred.size()) * 0.4 / (pred.size()[1] - 1)
        for i in range(pred.size()[0]): true[i, int(y[i])] = 0.6
        loss = criterion(pred, true)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss / len(dataloader)
    return avg_loss

def test(dataloader, model, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            true = torch.ones(pred.size()) * 0.4 / (pred.size()[1] - 1)
            for i in range(pred.size()[0]): true[i, int(y[i])] = 0.6
            test_loss += criterion(pred, true).item()
    avg_loss = test_loss / len(dataloader)
    return avg_loss

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--train_data', type=str, default='emnist')
    parser.add_argument('--output_dir', type=str, default='model/')
    parser.add_argument('--train_config', type=str, default='network/train.config')
    parser.add_argument('--network_file', type=str, default='network/layers.json')
    args = parser.parse_args()

    # Configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
            args.network_file = os.path.join(args.output_dir, os.path.basename(args.network_file))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.train_config, os.path.join(args.output_dir, 'train.config'))
        shutil.copy(args.network_file, os.path.join(args.output_dir, 'layers.json'))
    model_file = os.path.join(args.output_dir, 'model.pth')
    loss_file  = os.path.join(args.output_dir, 'loss.png')

    # Read training parameters
    if args.train_config is None:
        parser.error('Train config file has to be specified.')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    optimizer = train_config.get('optimizer', 'optimizer')
    learning_rate = train_config.getfloat('optimizer', 'learning_rate')
    momentum = train_config.getfloat('optimizer', 'momentum')
    betas = [float(x) for x in train_config.get('optimizer', 'betas').split(', ')]
    epsilon = train_config.getfloat('optimizer', 'epsilon')
    weight_decay = train_config.getfloat('optimizer', 'weight_decay')
    loss_func = train_config.get('training', 'loss')
    batch_size_train = train_config.getint('training', 'batch_size_train')
    batch_size_test = train_config.getint('training', 'batch_size_test')
    epochs = train_config.getint('training', 'epochs')

    # Set up dataset
    train_loader = setup_loader(args.train_data, batch_size=batch_size_train, train=True)
    test_loader = setup_loader(args.train_data, batch_size=batch_size_test, test=True)

    # Build the neural network
    with open(args.network_file) as file:
         layer_args = json.load(file)
         model = NeuralNet(layer_args)

    # Define loss function
    if loss_func == 'mse':
        criterion = nn.MSELoss()
    elif loss_func == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_func == 'nll':
        criterion = nn.NLLLoss()
    else:
        raise ValueError('Unknown loss function! Please choose mse, cross_entropy, or nll.')

    # Define optimizer
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                        betas=betas, eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer! Please choose sgd or adam.')

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Run training process over several epochs
    train_loss, test_loss = [], []
    for t in tqdm(range(epochs)):
        train_loss += [train(train_loader, model, criterion, optimizer, device)]
        test_loss  += [test(test_loader, model, criterion, device)]

    # Save plot of loss over epochs
    plt.plot(train_loss, '-b', label='Training')
    plt.plot(test_loss, '-r', label='Evaluation')
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss Across Batches')
    plt.title('Average Training and Evaluation Loss')
    plt.savefig(loss_file)

    # Save trained model
    torch.save(model.state_dict(), model_file)

if __name__ == "__main__":
    main()
