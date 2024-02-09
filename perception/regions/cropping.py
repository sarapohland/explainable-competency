import time
import numpy as np
import configparser
import itertools

import torch
import torch.nn.functional as F
torch.manual_seed(0)

from perception.regions.utils import RegionalCompetency

class Cropping(RegionalCompetency):
    def __init__(self, config, model, estimator):
        super(Cropping, self).__init__(config, model, estimator)

        # Read config file
        super(Cropping, self).read_config(config)
        self.read_config(config)

    def read_config(self, config_file):
        config = configparser.RawConfigParser()
        config.read(config_file)
        self.cell_height = config.getint('cropping', 'height')
        self.cell_width = config.getint('cropping', 'width')
        self.scale = config.get('cropping', 'scale')
        try:
            self.thresh = config.getfloat('cropping', 'thresh')
        except:
            self.thresh = None

    def get_regions(self, data, debug=False):
        if self.regions is not None:
            return self.regions

        # Partition image into cells and crop
        cropped_imgs = []
        img_height, img_width = data.size()[2], data.size()[3]
        for start_x in range(0, img_height-1, self.cell_height):
            for start_y in range(0, img_width-1, self.cell_width):
                # Crop image
                end_x = (start_x + self.cell_height) if (start_x + self.cell_height) < img_height else img_height
                end_y = (start_y + self.cell_width)  if (start_y + self.cell_width)  < img_width  else img_width
                cropped_img = data[:, :, start_x:end_x, start_y:end_y].clone()

                # Resize cropped image
                cropped_img = self.resize_image(cropped_img, img_height, img_width)
                cropped_imgs.append(cropped_img)

                # Visualize cropped image
                if debug:
                    import matplotlib.pyplot as plt
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)[0,:,:,:])
                    plt.title('Input Image')
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.swapaxes(np.swapaxes(cropped_img, 1, 2), 2, 3)[0,:,:,:])
                    plt.title('Cropped Portion of Image')
                    plt.show()
        
        # Compute competency for each cropped image
        cropped_imgs = torch.vstack(cropped_imgs)
        out = self.model(cropped_imgs)
        features = self.model.get_feature_vector(cropped_imgs)
        comps = self.estimator.comp_score(features, out)

        # Create figure displaying regional incompetency
        idx = 0
        crop_incomp = torch.zeros(img_height, img_width)
        for start_x in range(0, img_height-1, self.cell_height):
            for start_y in range(0, img_width-1, self.cell_width):
                end_x = (start_x + self.cell_height) if (start_x + self.cell_height) < img_height else img_height
                end_y = (start_y + self.cell_width)  if (start_y + self.cell_width)  < img_width  else img_width
                crop_incomp[start_x:end_x, start_y:end_y] = 1 - comps[idx]
                pixel_ranges = [range(start_x, end_x), range(start_y, end_y)]
                all_pixels = list(itertools.product(*pixel_ranges))
                idx += 1

        # Resize incompetency figure
        crop_incomp = self.resize_image(crop_incomp)

        # Scale predicted values
        self.regions = self.scale_values(crop_incomp, method=self.scale)
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
    parser.add_argument('--output_dir', type=str, default='results/perturb/')
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

    # Evaluate competency of cropped regions for each image in the test dataset
    cropping = Cropping(args.config_file, model, alice)
    for batch, (data, labels) in enumerate(test_loader):
        # Compute competency prediction
        cropping.comp = None
        cropping.regions = None
        comp = cropping.compute_competency(data)
        regions = cropping.get_regions(data, args.debug)

        # Plot highlighted region of image
        file = os.path.join(args.output_dir, '{}.png'.format(batch))
        title = 'Partitioning/Cropping Approach'
        plot_regions(data, regions, file, title)