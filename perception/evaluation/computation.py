import os
import json
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perception.network.model import NeuralNet
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.setup_dataloader import setup_loader

from perception.regions.cropping import Cropping
from perception.regions.masking import Masking
from perception.regions.perturbation import Perturbation
from perception.regions.gradients import Gradients
from perception.regions.reconstruction import Reconstruction

def main():
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--autoencoder_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/comp_map/')
    args = parser.parse_args()

    # Create folder to save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
    cropping  = Cropping(args.config_file, model, alice)
    masking   = Masking(args.config_file, model, alice)
    perturb   = Perturbation(args.config_file, model, alice)
    gradients = Gradients(args.config_file, model, alice)
    reconstr  = Reconstruction(args.config_file, model, alice, constructor)

    times = {'Cropping': [], 'Segmentation': [], 'Perturbation': [], 'Gradients': [], 'Reconstruction': []}
    for batch, (data, labels) in enumerate(test_loader):

        # Compute competency prediction
        gradients.comp = None
        comp = gradients.compute_competency(data)
        cropping.comp = masking.comp = perturb.comp = reconstr.comp = comp
        cropping.regions = masking.regions = perturb.regions = gradients.regions = reconstr.regions = None

        # Get regional competency estimates
        start = time.time()
        crop_regions = cropping.get_regions(data)
        times['Cropping'].append(time.time() - start)

        start = time.time()
        mask_regions = masking.get_regions(data)
        times['Segmentation'].append(time.time() - start)

        start = time.time()
        pert_regions = perturb.get_regions(data)
        times['Perturbation'].append(time.time() - start)

        start = time.time()
        grad_regions = gradients.get_regions(data)
        times['Gradients'].append(time.time() - start)

        start = time.time()
        reco_regions = reconstr.get_regions(data)
        times['Reconstruction'].append(time.time() - start)

    # Plot computation times
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame.from_dict(times)
    methods = ['Reconstruction', 'Gradients', 'Perturbation', 'Segmentation', 'Cropping']
    bp = df.boxplot(vert=False, column=methods)
    plt.title('Computation Times')
    plt.xlabel('Time (seconds)')
    plt.savefig(os.path.join(args.output_dir, 'times.png'), bbox_inches='tight')

    # Print average computation times
    for method in methods:
        print('{} average time: {}'.format(method, np.mean(times[method])))

if __name__=="__main__":
    main()