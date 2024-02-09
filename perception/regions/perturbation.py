import time
import numpy as np
import configparser

import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
torch.manual_seed(0)

from PIL import ImageFilter

from perception.regions.utils import RegionalCompetency

class Perturbation(RegionalCompetency):
    def __init__(self, config, model, estimator):
        super(Perturbation, self).__init__(config, model, estimator)

        # Read config file
        super(Perturbation, self).read_config(config)
        self.read_config(config)

    def read_config(self, config_file):
        config = configparser.RawConfigParser()
        config.read(config_file)
        self.fill = config.get('perturbation', 'value')
        self.scale = config.get('perturbation', 'scale')
        try:
            self.thresh = config.getfloat('perturbation', 'thresh')
        except:
            self.thresh = None

    def mask_image(self, img):
        if self.fill == 'zero':
            return torch.zeros_like(img)
        elif self.fill == 'one':
            return torch.ones_like(img)
        elif self.fill == 'average':
            img = np.squeeze(img.numpy())
            avg = np.median(img, axis=(1,2))
            img = np.full_like(img, avg[:,np.newaxis,np.newaxis])
            return torch.from_numpy(img)[None,:,:,:]
        elif self.fill == 'uniform':
            img = np.squeeze(img.numpy())
            min, max = np.min(img, axis=(1,2)), np.max(img, axis=(1,2))
            img = np.random.uniform(min[:,np.newaxis,np.newaxis], max[:,np.newaxis,np.newaxis], size=np.shape(img))
            return torch.from_numpy(img).float()[None,:,:,:]
        elif self.fill == 'gaussian':
            img = np.squeeze(img.numpy())
            avg, std = np.mean(img, axis=(1,2)), np.std(img, axis=(1,2))
            img = np.random.normal(avg[:,np.newaxis,np.newaxis], std[:,np.newaxis,np.newaxis], size=np.shape(img))
            img = np.clip(img, 0, 1)
            return torch.from_numpy(img).float()[None,:,:,:]
        elif self.fill == 'noise':
            img = np.squeeze(img.numpy())
            noise = np.random.normal(0, 0.1, size=np.shape(img))
            noisy = img + noise
            noisy = np.clip(noisy, 0, 1)
            return torch.from_numpy(noisy).float()[None,:,:,:]
        elif self.fill == 'blur':
            shape = list(img.size())
            img = torch.squeeze(img)
            pil_img = to_pil_image(img)
            blur_img = pil_img.filter(ImageFilter.BLUR)
            return torch.reshape((pil_to_tensor(blur_img) / 255).float(), shape)
        else:
            raise NotImplementedError('Unknown fill type for perturbation method.')

    def get_regions(self, data, debug=False):
        if self.regions is not None:
            return self.regions

        img_height, img_width = data.size()[2], data.size()[3]

        # Perform image segmentation
        resized = self.resize_image(data)
        self.pixels = self.segment_image(resized)

        perturbed_imgs, corresp_pixels = [], []
        for pixels in self.pixels:
            # "Remove" region of interest
            masked_img = self.mask_image(resized).clone()
            perturbed_img = resized.clone()
            perturbed_img[:, :, pixels[0, :], pixels[1, :]] = masked_img[:, :, pixels[0, :], pixels[1, :]].clone()

            # Resize perturbed image
            perturbed_img = self.resize_image(perturbed_img, img_height, img_width)
            perturbed_imgs.append(perturbed_img)

            # Visualize perturbed image
            if debug:
                import matplotlib.pyplot as plt
                plt.subplot(1, 2, 1)
                plt.imshow(np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)[0,:,:,:])
                plt.title('Input Image')
                plt.subplot(1, 2, 2)
                plt.imshow(np.swapaxes(np.swapaxes(perturbed_img, 1, 2), 2, 3)[0,:,:,:])
                plt.title('Perturbed Image')
                plt.show()

        # Compute competency for each perturbed image
        perturbed_imgs = torch.vstack(perturbed_imgs)
        out = self.model(perturbed_imgs)
        features = self.model.get_feature_vector(perturbed_imgs)
        comps = self.estimator.comp_score(features, out)

        # Create figure displaying regional incompetency
        seg_comp = torch.zeros(self.height, self.width)
        for pixels, comp in zip(self.pixels, comps):
            mask = torch.zeros(self.height, self.width)
            mask[pixels[0, :], pixels[1, :]] = 1
            seg_comp += mask * torch.maximum(comp, self.comp)
        self.values = list(torch.maximum(comps, self.comp))

        # Scale predicted values
        self.regions = self.scale_values(seg_comp, method=self.scale)
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

    # Evaluate competency of perturbed images in the test dataset
    perturb = Perturbation(args.config_file, model, alice)
    for batch, (data, labels) in enumerate(test_loader):
        # Compute competency prediction
        perturb.comp = None
        perturb.regions = None
        comp = perturb.compute_competency(data)
        regions = perturb.get_regions(data, args.debug)

        # Plot highlighted region of image
        file = os.path.join(args.output_dir, '{}.png'.format(batch))
        title = 'Pixel Perturbation Approach'
        plot_regions(data, regions, file, title)