import os
import time
import json
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from perception.network.model import NeuralNet
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.setup_dataloader import setup_loader

from perception.regions.cropping import Cropping
from perception.regions.masking import Masking
from perception.regions.perturbation import Perturbation
from perception.regions.gradients import Gradients
from perception.regions.reconstruction import Reconstruction
from perception.regions.ensemble import Ensemble

from perception.evaluation.accuracy import get_accuracy


def main():
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--autoencoder_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--thresh', type=float, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Create folder to save results
    output_dir = 'results/{}/'.format(args.test_data)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load trained classification model
    with open(args.model_dir + 'layers.json') as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(args.model_dir + 'model.pth'))

    # Load pretrained ALICE model
    alice = pickle.load(open(args.model_dir + 'alice.p', 'rb'))

    # Load trained reconstruction model
    encoder_args = json.load(open(os.path.join(args.autoencoder_dir, 'encoder.json')))
    decoder_args = json.load(open(os.path.join(args.autoencoder_dir, 'decoder.json')))
    constructor = AutoEncoder(encoder_args, decoder_args)
    constructor.load_state_dict(torch.load(os.path.join(args.autoencoder_dir, 'model.pth')))

    # Create data loader
    test_loader = setup_loader(args.test_data, batch_size=1, ood=True)

    # Instantiate all of the regional competency approaches
    masking   = Masking(args.config_file, model, alice)
    perturb   = Perturbation(args.config_file, model, alice)
    gradients = Gradients(args.config_file, model, alice)
    reconstr  = Reconstruction(args.config_file, model, alice, constructor)

    # Instantiate ensembling methods
    mp = Ensemble([masking, perturb], [1, 1])
    mg = Ensemble([masking, gradients], [1, 1])
    mr = Ensemble([masking, reconstr], [1, 1])
    pg = Ensemble([perturb, gradients], [1, 1])
    pr = Ensemble([perturb, reconstr], [1, 1])
    gr = Ensemble([gradients, reconstr], [1, 1])
    mpg = Ensemble([masking, perturb, gradients], [1, 1, 1])
    mpr = Ensemble([masking, perturb, reconstr], [1, 1, 1])
    mgr = Ensemble([masking, gradients, reconstr], [1, 1, 1])
    pgr = Ensemble([perturb, gradients, reconstr], [1, 1, 1])
    mpgr = Ensemble([masking, perturb, gradients, reconstr], [1, 1, 1, 1])
    ensembles = [mp, mg, mr, pg, pr, gr, mpg, mpr, mgr, pgr, mpgr]
    methods = ['M+P', 'M+G', 'M+R', 'P+G', 'P+R', 'G+R', 'M+P+G', 'M+P+R', 'M+G+R', 'P+G+R', 'M+P+G+R']

    # Load true segmentation labels
    label_dir = 'data/{}/'.format(args.test_data)
    segmentation = pickle.load(open(os.path.join(label_dir, 'segmentation.p'), 'rb'))
    seg_pixels = segmentation['pixels']
    seg_labels = segmentation['labels']

    acc = {'M+P': [], 'M+G': [], 'M+R': [], 'P+G': [], 'P+R': [], 'G+R': [], 'M+P+G': [], 'M+P+R': [], 'M+G+R': [], 'P+G+R': [], 'M+P+G+R': []}
    tpr = {'M+P': [], 'M+G': [], 'M+R': [], 'P+G': [], 'P+R': [], 'G+R': [], 'M+P+G': [], 'M+P+R': [], 'M+G+R': [], 'P+G+R': [], 'M+P+G+R': []}
    tnr = {'M+P': [], 'M+G': [], 'M+R': [], 'P+G': [], 'P+R': [], 'G+R': [], 'M+P+G': [], 'M+P+R': [], 'M+G+R': [], 'P+G+R': [], 'M+P+G+R': []}
    ppv = {'M+P': [], 'M+G': [], 'M+R': [], 'P+G': [], 'P+R': [], 'G+R': [], 'M+P+G': [], 'M+P+R': [], 'M+G+R': [], 'P+G+R': [], 'M+P+G+R': []}
    npv = {'M+P': [], 'M+G': [], 'M+R': [], 'P+G': [], 'P+R': [], 'G+R': [], 'M+P+G': [], 'M+P+R': [], 'M+G+R': [], 'P+G+R': [], 'M+P+G+R': []}
    
    for batch, (data, _) in enumerate(test_loader):

        # Get true labels of image
        true_labels = seg_labels[batch]
        true_regions = np.zeros((masking.height, masking.width))
        for pixels, label in zip(seg_pixels[batch], true_labels):
            true_regions[pixels[0, :], pixels[1, :]] = label

        # Skip analysis if we do not have good labels
        if not np.any(true_regions == 1):
            continue

        # Compute competency prediction
        gradients.comp = None
        gradients.gradients = []
        gradients.activations = []
        comp = gradients.compute_competency(data)
        masking.comp = perturb.comp = reconstr.comp = comp
        masking.regions = perturb.regions = gradients.regions = reconstr.regions = None

        # Get regional competency estimates
        regions = [ensemble.get_regions(data) for ensemble in ensembles]

        # Highlight regions above threshold
        regions_thresh = []
        for ensemble, region in zip(ensembles, regions):
            if args.thresh is not None:
                thresh = args.thresh
            else:
                thresholds = [method.thresh for method in ensemble.methods]
                thresh = np.mean(thresholds)
            region_thresh = np.zeros_like(region)
            region_thresh[region > thresh] = 1
            regions_thresh.append(region_thresh)

        # Compute accuracy
        for region, method in zip(regions_thresh, methods):
            accuracy = get_accuracy(true_regions, region)
            acc[method].append(accuracy['ACC'])
            tpr[method].append(accuracy['TPR'])
            tnr[method].append(accuracy['TNR'])
            ppv[method].append(accuracy['PPV'])
            npv[method].append(accuracy['NPV'])

        if args.debug:
            fig = plt.figure(figsize=(12, 8))

            plt.subplot(2, 4, 1)
            data = np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)
            im = plt.imshow(data[0,:,:,:])
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(2, 4, 2)
            im = plt.imshow(true_regions, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('True Segmentation Labels')
            plt.axis('off')

            for fig, (region, method) in enumerate(zip(regions_thresh, methods)):
                if fig > 5:
                    break
                plt.subplot(2, 4, fig + 3)
                im = plt.imshow(region, cmap='coolwarm', vmin=0, vmax=1)
                plt.title('{} Accuracy: {}'.format(method, round(acc[method][-1].item(), 3)))
                plt.axis('off')

            plt.tight_layout()
            plt.suptitle('Competency Score: {}'.format(round(comp.item(), 3)), size='x-large')
            plt.show()

    # Plot accuracy of estimates
    plt.figure(figsize=(8, 8))
    df = pd.DataFrame.from_dict(acc)
    bp = df.boxplot(vert=False, column=methods.reverse())
    plt.title('Accuracy of Ensembling Methods')
    plt.xlabel('Method Accuracy')
    plt.savefig(os.path.join(output_dir, 'ensemble.png'), bbox_inches='tight')

    # Print average accuracies
    for method in methods:
        print(method)
        print('Avg ACC: {}'.format(np.nanmean(acc[method])))
        print('Avg TPR: {}'.format(np.nanmean(tpr[method])))
        print('Avg TNR: {}'.format(np.nanmean(tnr[method])))
        print('Avg PPV: {}'.format(np.nanmean(ppv[method])))
        print('Avg NPV: {}'.format(np.nanmean(npv[method])))

if __name__=="__main__":
    main()