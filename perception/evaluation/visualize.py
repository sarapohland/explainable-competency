import os
import json
import torch
import pickle
import argparse
import numpy as np
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
    parser.add_argument('--output_dir', type=str, default='results/compare/')
    parser.add_argument('--threshold', action='store_true')
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

    for batch, (data, labels) in enumerate(test_loader):
        # Compute competency prediction
        gradients.comp = None
        gradients.gradients = []
        gradients.activations = []
        comp = gradients.compute_competency(data)
        cropping.comp = masking.comp = perturb.comp = reconstr.comp = comp
        cropping.regions = masking.regions = perturb.regions = gradients.regions = reconstr.regions = None

        # Get regional competency estimates
        grad_regions = gradients.get_regions(data)
        crop_regions = cropping.get_regions(data)
        mask_regions = masking.get_regions(data)
        pert_regions = perturb.get_regions(data)
        reco_regions = reconstr.get_regions(data)

        # Highlight regions above threshold
        if args.threshold:
            crop_regions[crop_regions <= cropping.thresh] = 0
            crop_regions[crop_regions >  cropping.thresh] = 1
            mask_regions[mask_regions <= masking.thresh] = 0
            mask_regions[mask_regions >  masking.thresh] = 1
            pert_regions[pert_regions <= perturb.thresh] = 0
            pert_regions[pert_regions >  perturb.thresh] = 1
            grad_regions[grad_regions <= gradients.thresh] = 0
            grad_regions[grad_regions >  gradients.thresh] = 1
            reco_regions[reco_regions <= reconstr.thresh] = 0
            reco_regions[reco_regions >  reconstr.thresh] = 1

        # Plot comparison of approaches
        fig = plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        data = np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)
        im = plt.imshow(data[0,:,:,:])
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        im = plt.imshow(crop_regions, cmap='coolwarm')
        plt.title('Cropping Approach')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        im = plt.imshow(mask_regions, cmap='coolwarm')
        plt.title('Segmentation Approach')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        im = plt.imshow(pert_regions, cmap='coolwarm')
        plt.title('Perturbation Approach')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        im = plt.imshow(grad_regions, cmap='coolwarm')
        plt.title('Gradients Approach')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        im = plt.imshow(reco_regions, cmap='coolwarm')
        plt.title('Reconstruction Approach')
        plt.axis('off')

        plt.tight_layout()
        # plt.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # plt.colorbar(im, cax=cbar_ax)
        # plt.suptitle('Competency Score: {}'.format(round(comp.item(), 3)), size='x-large')
        plt.suptitle('Dependence of Incompetency on Pixel Values')
        plt.savefig(os.path.join(args.output_dir, '{}.png'.format(batch)))
        plt.close()

if __name__=="__main__":
    main()