import time
import numpy as np
import configparser

import torch
import torch.nn.functional as F
torch.manual_seed(0)

from perception.regions.utils import RegionalCompetency

class Gradients(RegionalCompetency):
    def __init__(self, config, model, estimator):
        super(Gradients, self).__init__(config, model, estimator)

        # Read config file
        super(Gradients, self).read_config(config)
        self.read_config(config)

        # Register hook on convolutional layer
        self.gradients = []
        self.activations = []
        self.register_hook(self.model)

    def read_config(self, config_file):
        config = configparser.RawConfigParser()
        config.read(config_file)
        self.layers = [int(x) for x in config.get('gradients', 'layers').split(', ')]
        self.clipping = config.get('gradients', 'clipping')
        self.scale = config.get('gradients', 'scale')
        try:
            self.thresh = config.getfloat('gradients', 'thresh')
        except:
            self.thresh = None

    def register_hook(self, model):
        model.eval()
        for layer in self.layers:
            model.net[layer].register_full_backward_hook(self.backward_hook, prepend=False)
            model.net[layer].register_forward_hook(self.forward_hook, prepend=False)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output)

    def forward_hook(self, module, args, output):
        self.activations.append(output)

    def get_heatmap(self, output, relu=False, absval=True):
        output.backward(retain_graph=False)
        self.activations = self.activations[:len(self.layers)]
        self.activations.reverse()

        heatmaps = []
        for gradients, activations in zip(self.gradients, self.activations):
            pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
            for i in range(activations.size()[1]):
                activations[:, i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activations, dim=1).squeeze()

            if self.clipping == 'relu':
                heatmap = F.relu(heatmap)
            elif self.clipping == 'abs':
                heatmap = torch.abs(heatmap)
            elif self.clipping == 'negabs':
                heatmap = -torch.abs(heatmap)
            elif self.clipping == 'none':
                pass
            else:
                raise NotImplementedError('Clipping function {} is not implemented for gradients.'.format(self.clipping))
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps)
        return heatmaps
    
    def get_regions(self, data, debug=False):
        if self.regions is not None:
            return self.regions

        # Compute competency gradients
        all_comp_grads = self.get_heatmap(self.comp)
        
        if debug:
            import matplotlib.pyplot as plt
            plt.subplot(1, len(self.layers)+1, 1)
            plt.imshow(np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)[0,:,:,:])
            plt.title('Input Image')
            for i, comp_grads in enumerate(all_comp_grads):
                plt.subplot(1, len(self.layers)+1, i+2)
                plt.imshow(comp_grads.detach(), cmap='coolwarm')
                plt.title('Gradients of Competency Score')
            plt.show()

        # Perform image segmentation
        data = self.resize_image(data)
        self.pixels = self.segment_image(data)

        # Compute competency gradients for segmented image
        self.values = []
        seg_grads = torch.zeros_like(all_comp_grads[0])
        for pixels in self.pixels:
            num_pixels = pixels.size()[1]
            mask = torch.zeros(self.height, self.width)
            mask[pixels[0, :], pixels[1, :]] = 1
            
            avg_grad = 0
            for comp_grads in all_comp_grads:
                comp_grads = (comp_grads - torch.min(all_comp_grads)) / (torch.max(all_comp_grads) - torch.min(all_comp_grads))
                avg_grad += torch.sum(comp_grads * mask) / num_pixels
            seg_grads += mask * avg_grad
            self.values.append(avg_grad.item())

        # Scale predicted values
        self.regions = self.scale_values(seg_grads, method=self.scale)
        return self.regions


if __name__=="__main__":
    import os
    import json
    import pickle
    import argparse

    from perception.network.model import NeuralNet
    from perception.datasets.setup_dataloader import setup_loader
    from perception.regions.utils import plot_regions

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/grads/')
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

    # Create data loader
    test_loader = setup_loader(args.test_data, batch_size=1, ood=True)

    # Evaluate gradients for each image in the test dataset
    gradients = Gradients(args.config_file, model, alice)
    for batch, (data, labels) in enumerate(test_loader):
        # Compute competency prediction
        gradients.comp = None
        gradients.regions = None
        gradients.gradients = []
        gradients.activations = []
        comp = gradients.compute_competency(data)
        regions = gradients.get_regions(data, args.debug)

        # Plot highlighted region of image
        file = os.path.join(args.output_dir, '{}.png'.format(batch))
        title = 'Pixel Gradient Approach'
        plot_regions(data, regions, file, title)
