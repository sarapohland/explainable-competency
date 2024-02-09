import os
import json
import shutil
import pickle
import argparse
import numpy as np
import configparser
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib.pyplot as plt

from perception.inpainting.utils import resize_image
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.custom_dataset import CustomDataset, SegmentedDataset

def test(dataloader, model, device, pretrain=False):
    model.eval()
    inputs, labels, preds = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            input = X.to(device)
            height, width = input.size()[2], input.size()[3]
            true  = input if pretrain else resize_image(y, height, width).to(device)
            pred  = model(input)
            inputs.append(input)
            labels.append(true)
            preds.append(pred)
    return torch.concatenate(inputs), torch.concatenate(labels), torch.concatenate(preds)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('test_data', type=str)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--model_dir', type=str, default='model/')
    args = parser.parse_args()

    # Load trained model
    encoder_args = json.load(open(os.path.join(args.model_dir, 'encoder.json')))
    decoder_args = json.load(open(os.path.join(args.model_dir, 'decoder.json')))
    model = AutoEncoder(encoder_args, decoder_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))

    # Create data loader
    datasets = pickle.load(open(args.test_data, 'rb'))
    ood_dataset  = datasets['ood']
    test_dataset = datasets['test']
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Get model predictions
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs, labels, preds = test(test_loader, model, device, pretrain=args.pretrain)

    # Save reconstructed images
    output_dir = os.path.join(args.model_dir, 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    loss_func = nn.MSELoss()
    for i, (input, label, pred) in enumerate(zip(inputs, labels, preds)):
        if i > 200:
            break 
        loss = loss_func(label, pred).item()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        label = np.float32(label.numpy())
        label = np.swapaxes(np.swapaxes(label, 0, 1), 1, 2)
        plt.imshow(label)
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        input = np.float32(input.numpy())
        input = np.swapaxes(np.swapaxes(input, 0, 1), 1, 2)
        plt.imshow(input)
        plt.title('Input/Masked Image')

        plt.subplot(1, 3, 3)
        pred = np.float32(pred.numpy())
        pred = np.swapaxes(np.swapaxes(pred, 0, 1), 1, 2)
        plt.imshow(pred)
        plt.title('Reconstructed Image')

        plt.suptitle('Reconstruction Loss: {}'.format(round(loss, 3)))
        file = os.path.join(output_dir, '{}.png'.format(i))
        plt.savefig(file)
        plt.close()

if __name__ == "__main__":
    main()
