import os
import json
import shutil
import pickle
import argparse
import numpy as np
import configparser
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from perception.inpainting.utils import resize_image
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.custom_dataset import CustomDataset, SegmentedDataset

def train(dataloader, model, criterion, optimizer, device, pretrain=False):
    model.train()
    train_loss = 0
    for X, y in dataloader:
        input = X.to(device)
        height, width = input.size()[2], input.size()[3]
        true  = input if pretrain else resize_image(y, height, width).to(device)
        pred  = model(input)
        loss  = criterion(pred, true)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss / len(dataloader)
    return avg_loss

def test(dataloader, model, criterion, device, pretrain=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            input = X.to(device)
            height, width = input.size()[2], input.size()[3]
            true  = input if pretrain else resize_image(y, height, width).to(device)
            pred  = model(input)
            test_loss += criterion(pred, true).item()
    avg_loss = test_loss / len(dataloader)
    return avg_loss

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('train_data', type=str)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--output_dir', type=str, default='model/')
    parser.add_argument('--train_config', type=str, default='inpainting/train.config')
    parser.add_argument('--encoder_file', type=str, default='inpainting/encoder.json')
    parser.add_argument('--decoder_file', type=str, default='inpainting/decoder.json')
    parser.add_argument('--init_model', type=str, default=None)
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
            args.encoder_file = os.path.join(args.output_dir, os.path.basename(args.encoder_file))
            args.decoder_file = os.path.join(args.output_dir, os.path.basename(args.decoder_file))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.train_config, os.path.join(args.output_dir, 'train.config'))
        shutil.copy(args.encoder_file, os.path.join(args.output_dir, 'encoder.json'))
        shutil.copy(args.decoder_file, os.path.join(args.output_dir, 'decoder.json'))
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
    datasets = pickle.load(open(args.train_data, 'rb'))
    train_dataset = datasets['train']
    test_dataset  = datasets['test']
    train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size_test,  shuffle=True)

    # Build the neural network
    encoder_args = json.load(open(args.encoder_file))
    decoder_args = json.load(open(args.decoder_file))
    model = AutoEncoder(encoder_args, decoder_args)

    if args.init_model is not None:
        model.load_state_dict(torch.load(args.init_model))

    # Define loss function
    if loss_func == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Unknown loss function! Only MSE is currently implemented.')

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
        train_loss += [train(train_loader, model, criterion, optimizer, device, pretrain=args.pretrain)]
        test_loss  += [test(test_loader, model, criterion, device, pretrain=args.pretrain)]

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
