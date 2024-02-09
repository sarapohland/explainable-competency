import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from perception.network.model import NeuralNet
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.setup_dataloader import setup_loader
from perception.regions.utils import RegionalCompetency

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    elif val in ('s', 'skip', '?', 'unknown'):
        return -1
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
def collect_labels(data, all_pixels):

    labels = []
    idx = 0
    while idx < len(all_pixels):
        pixels = all_pixels[idx]

        # Create a mask for the current segment
        color = torch.FloatTensor([0, 0, 1])[None,:,None,None]
        masked_img = color.expand(data.size()).clone()
        masked_img[:, :, pixels[0, :], pixels[1, :]] = data[:, :, pixels[0, :], pixels[1, :]].clone()

        # Visualize masked image
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)[0,:,:,:])
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Region of Interest')
        plt.imshow(np.swapaxes(np.swapaxes(masked_img, 1, 2), 2, 3)[0,:,:,:])
        plt.show(block=False)

        # Listen for user input
        choice = input('Does this segment contain a structure not present in the training set? ')
        try:
            # Save label of segment
            label = strtobool(choice)
            labels.append(label)
            idx += 1
        except:
            try:
                del labels[-1]
                idx -= 1
            except:
                pass
        plt.close()
    
    return labels

def display_labels(data, true_regions):
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    data = np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)
    im = plt.imshow(data[0,:,:,:])
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    true_regions[true_regions == -1] = 0.5
    im = plt.imshow(true_regions, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('True Segmentation Labels')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)

def main():

    # Create folder to save results
    output_dir = 'data/{}/'.format(args.test_data)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load trained classification model
    with open(args.model_dir + 'layers.json') as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(args.model_dir + 'model.pth'))

    # Load pretrained ALICE model
    alice = pickle.load(open(args.model_dir + 'alice.p', 'rb'))

    # Create data loader
    test_loader = setup_loader(args.test_data, batch_size=1, ood=True)

    # Instantiate base regional competency method
    regions = RegionalCompetency(args.config_file, model, alice)
    regions.read_config(args.config_file)

    # Load existing segmentation labels (if available)
    try:
        segmentation = pickle.load(open(os.path.join(output_dir, 'segmentation.p'), 'rb'))
        seg_pixels = segmentation['pixels']
        seg_labels = segmentation['labels']
    except:
        seg_pixels, seg_labels = [], []

    # Create labels for segmented regions
    for batch, (data, labels) in enumerate(test_loader):
        # Skip data samples that have already been labeled
        if batch < len(seg_pixels):
            continue
        print('Test case: ', batch)

        # Perform image segmentation
        data = regions.resize_image(data)
        all_pixels = regions.segment_image(data)

        # Get segment labels from user
        labels = collect_labels(data, all_pixels)
        seg_labels.append(torch.IntTensor(labels))
        seg_pixels.append(all_pixels)
    
        # Allow user to correct their labels
        correct = True
        while correct:
            # Display true labels of segments
            true_regions = np.zeros((data.size()[2], data.size()[3]))
            for pixels, label in zip(seg_pixels[batch], seg_labels[batch]):
                true_regions[pixels[0, :], pixels[1, :]] = label
            display_labels(data, true_regions)

            # Determine if labels need to be corrected
            choice = input('Does this image need to be corrected? ')
            correct = strtobool(choice)

            # Collect new labels if necessary
            if correct:
                labels = collect_labels(data, seg_pixels[batch])
                seg_labels[batch] = labels
            plt.close()
    
        # Save segments
        segmentation = {'pixels': seg_pixels, 'labels': seg_labels}
        pickle.dump(segmentation, open(os.path.join(output_dir, 'segmentation.p'), 'wb'))

def test():

    output_dir = 'data/{}/'.format(args.test_data)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load trained classification model
    with open(args.model_dir + 'layers.json') as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(args.model_dir + 'model.pth'))

    # Load pretrained ALICE model
    alice = pickle.load(open(args.model_dir + 'alice.p', 'rb'))

    # Instantiate base regional competency method
    regions = RegionalCompetency(args.config_file, model, alice)
    regions.read_config(args.config_file)

    # Load true segmentation labels
    segmentation = pickle.load(open(os.path.join(output_dir, 'segmentation.p'), 'rb'))
    seg_pixels = segmentation['pixels']
    seg_labels = segmentation['labels']

    # Create data loader
    test_loader = setup_loader(args.test_data, batch_size=1, ood=True)

    for batch, (data, _) in enumerate(test_loader):
        if batch < args.start_idx:
            continue
        print('Test case: ', batch)

        correct = True
        while correct:
            # Display true labels of segments
            true_regions = np.zeros((regions.height, regions.width))
            for pixels, label in zip(seg_pixels[batch], seg_labels[batch]):
                true_regions[pixels[0, :], pixels[1, :]] = label
            display_labels(data, true_regions)

            # Determine if labels need to be corrected
            choice = input('Does this image need to be corrected? ')
            correct = strtobool(choice)

            # Collect new labels if necessary
            if correct:
                labels = collect_labels(regions.resize_image(data), seg_pixels[batch])
                seg_labels[batch] = labels
            plt.close()

        # Save segments
        segmentation = {'pixels': seg_pixels, 'labels': seg_labels}
        pickle.dump(segmentation, open(os.path.join(output_dir, 'segmentation.p'), 'wb'))

if __name__=="__main__":

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()

    test() if args.test else main()
       