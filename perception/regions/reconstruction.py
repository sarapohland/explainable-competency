import time
import numpy as np
import configparser

import torch
import torch.nn as nn
torch.manual_seed(0)

from perception.regions.utils import RegionalCompetency

class Reconstruction(RegionalCompetency):
    def __init__(self, config, model, estimator, const):
        super(Reconstruction, self).__init__(config, model, estimator)

        # Read config file
        super(Reconstruction, self).read_config(config)
        self.read_config(config)

        # Set autoencoder
        self.const = const
        self.const.eval()

    def read_config(self, config_file):
        config = configparser.RawConfigParser()
        config.read(config_file)
        self.method = config.get('reconstruction', 'method')
        self.loss = config.get('reconstruction', 'loss')
        self.scale = config.get('reconstruction', 'scale')
        try:
            self.thresh = config.getfloat('reconstruction', 'thresh')
        except:
            self.thresh = None

    def inpainting_loss(self, input_img, debug=False):
        masked_imgs = []
        for pixels in self.pixels:
            # Create a mask tensor for the current segment
            masked_img = input_img.clone()
            masked_img[:, :, pixels[0, :], pixels[1, :]] = 1
            masked_imgs.append(masked_img)

        # Reconstruct missing segments of image
        masked_imgs = torch.vstack(masked_imgs)
        pred_imgs = self.const(masked_imgs)
        reconst_imgs = input_img.expand(pred_imgs.size()).clone()
        for i, pixels in enumerate(self.pixels):
            reconst_imgs[i, :, pixels[0, :], pixels[1, :]] = pred_imgs[i, :, pixels[0, :], pixels[1, :]].clone()

        # Compute reconstruction loss per segment
        target_imgs = input_img.expand(pred_imgs.size()).clone()
        if self.loss == 'mse':
            loss_func = nn.MSELoss(reduction='none')
            losses = loss_func(target_imgs, reconst_imgs)
            seg_losses = torch.mean(losses, dim=(1,2,3)).detach()
        elif self.loss == 'l1':
            loss_func = nn.L1Loss(reduction='none')
            losses = loss_func(target_imgs, reconst_imgs)
            seg_losses = torch.mean(losses, dim=(1,2,3)).detach()
        elif self.loss == 'kldiv':
            loss_func = nn.KLDivLoss(reduction='none')
            losses = loss_func(target_imgs, reconst_imgs)
            seg_losses = torch.mean(losses, dim=(1,2,3)).detach()
        elif self.loss == 'cos':
            loss_func = nn.CosineSimilarity()
            losses = loss_func(target_imgs, reconst_imgs)
            seg_losses = torch.mean(losses, dim=(1,2)).detach()
            # seg_losses = 1 / seg_losses
        elif self.loss == 'dist':
            loss_func = nn.PairwiseDistance(p=1, keepdim=True)
            segs, chan, height, width = target_imgs.size()
            losses = torch.zeros((segs, height, width))
            for i in range(height):
                for j in range(width):
                    losses[:,i,j] = loss_func(target_imgs[:,:,i,j], reconst_imgs[:,:,i,j]).flatten()
            seg_losses = torch.mean(losses, dim=(1,2)).detach()
        else:
            raise NotImplementedError('{} loss function is not implemented for inpainting method.'.format(self.loss))
        
        # Normalize loss based on size of segment
        seg_pixels = [pixels.size()[1] for pixels in self.pixels]
        tot_pixels = self.height * self.width
        seg_losses /= (torch.Tensor(seg_pixels) / tot_pixels)

        # Invert similarity measure
        if self.loss == 'cos':
            seg_losses = 1 / seg_losses

        # Visualize autoencoder reconstruction
        if debug:
            import matplotlib.pyplot as plt
            for masked_img, reconst_img, loss in zip(masked_imgs, reconst_imgs, seg_losses):
                plt.subplot(1, 3, 1)
                plt.imshow(np.swapaxes(np.swapaxes(input_img[0], 0, 1), 1, 2))
                plt.title('Input Image')
                plt.subplot(1, 3, 2)
                plt.imshow(np.swapaxes(np.swapaxes(masked_img, 0, 1), 1, 2))
                plt.title('Masked Image')
                plt.subplot(1, 3, 3)
                plt.imshow(np.swapaxes(np.swapaxes(reconst_img.detach(), 0, 1), 1, 2))
                plt.title('Reconstructed Image')
                plt.suptitle('Reconstruction Loss: {}'.format(round(loss.item(), 3)))
                plt.show()

        return seg_losses
    
    def averaging_loss(self, input_img, debug=False):
        # Reconstruct image
        pred_img = self.const(input_img)

        # Compute reconstruction loss per segment
        if self.loss == 'mse':
            loss_func = nn.MSELoss(reduction='none')
            losses = loss_func(input_img, pred_img)
            seg_losses = torch.Tensor([torch.mean(losses[:, :, pixels[0, :], pixels[1, :]]) for pixels in self.pixels])
            losses = torch.mean(losses, dim=(0,1)).detach().numpy()
        elif self.loss == 'cos':
            loss_func = nn.CosineSimilarity()
            losses = loss_func(input_img, pred_img)
            seg_losses = torch.Tensor([torch.mean(losses[:, pixels[0, :], pixels[1, :]]) for pixels in self.pixels])
            losses = torch.mean(losses, dim=0).detach().numpy()
        else:
            raise NotImplementedError('{} loss function is not implemented for averaging method.'.format(self.loss))

        # Invert similarity measure
        if self.loss == 'cos':
            seg_losses = 1 / seg_losses

        # Visualize autoencoder reconstruction
        if debug:
            import matplotlib.pyplot as plt
            plt.subplot(1, 3, 1)
            plt.imshow(np.swapaxes(np.swapaxes(input_img[0], 0, 1), 1, 2))
            plt.title('Input Image')
            plt.subplot(1, 3, 2)
            plt.imshow(np.swapaxes(np.swapaxes(pred_img[0].detach(), 0, 1), 1, 2))
            plt.title('Reconstructed Image')
            plt.subplot(1, 3, 3)
            plt.imshow(losses)
            plt.title('Reconstructed Loss')
            plt.show()

        return torch.FloatTensor(seg_losses)

    def get_regions(self, data, debug=False):
        if self.regions is not None:
            return self.regions

        # Perform image segmentation
        resized = self.resize_image(data)
        self.pixels = self.segment_image(resized)

        # Compute reconstruction loss
        if self.method == 'inpainting':
            losses = self.inpainting_loss(resized, debug)
        elif self.method == 'averaging':
            losses = self.averaging_loss(resized, debug)
        else:
            raise NotImplementedError('{} method has not been implemented for reconstruction.'.format(self.method))

        # Create image of reconstruction losses
        loss_img = torch.zeros(self.height, self.width)
        for pixels, loss in zip(self.pixels, losses):
            mask = torch.zeros(self.height, self.width)
            mask[pixels[0, :], pixels[1, :]] = 1
            loss_img += mask * loss
        self.values = list(losses.detach().numpy())

        # Scale predicted values
        self.regions = self.scale_values(loss_img, method=self.scale)
        return self.regions


if __name__=="__main__":
    import os
    import json
    import pickle
    import argparse

    from perception.network.model import NeuralNet
    from perception.inpainting.autoencoder import AutoEncoder
    from perception.datasets.setup_dataloader import setup_loader
    from perception.regions.utils import plot_regions

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--autoencoder_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/reconstr/')
    parser.add_argument('--debug', action='store_true')
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

    # Compute reconstruction loss for each image in the test dataset
    reconstruction = Reconstruction(args.config_file, model, alice, constructor)
    for batch, (data, labels) in enumerate(test_loader):
        # Compute competency prediction
        reconstruction.comp = None
        reconstruction.regions = None
        comp = reconstruction.compute_competency(data)
        regions = reconstruction.get_regions(data, args.debug)

        # Plot highlighted region of image
        file = os.path.join(args.output_dir, '{}.png'.format(batch))
        title = 'Reconstruction Loss Approach'
        plot_regions(data, regions, file, title)
