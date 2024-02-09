import time
import torch
import numpy as np
import configparser
import matplotlib.pyplot as plt

from scipy import stats

from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from perception.competency.alice import Alice
from perception.regions.segment import segment, segment_pixels

class RegionalCompetency():
    def __init__(self, config, model, estimator):
        # Set model and competency estimator
        self.model = model
        self.model.eval()
        self.estimator = estimator

        # Initialize competency and regional values
        self.comp = None
        self.regions = None

    def read_config(self, config_file):
        config = configparser.RawConfigParser()
        config.read(config_file)
        self.height = config.getint('size', 'height')
        self.width = config.getint('size', 'width')
        self.sigma = config.getfloat('segmentation', 'sigma')
        self.seg_scale = config.getfloat('segmentation', 'scale')
        self.min_size = config.getint('segmentation', 'min_size')

    def compute_competency(self, data):
        # Compute model outputs
        out = self.model(data)

        # Compute competency predictions
        if isinstance(self.estimator, Alice):
            features = self.model.get_feature_vector(data)
            comp = self.estimator.comp_score(features, out)
        else:
            comp = self.estimator.comp_score(data, out)
        
        if self.comp is None:
            self.comp = comp
        return comp
    
    def resize_image(self, data, height=None, width=None):
        # Resize image
        height = height if height is not None else self.height
        width  = width  if width  is not None else self.width

        if isinstance(data, np.ndarray):
            type = data.dtype
            shape = list(np.shape(data))
            shape[-2], shape[-1] = height, width
            data = np.squeeze(data)
            pil_img = Image.fromarray(data)
            pil_img = pil_img.resize((width, height))
            np_img = np.array(pil_img)
            if type == 'float32':
                np_img = (np_img / 255).astype(type)
            return np.reshape(np_img, shape)
        
        elif isinstance(data, torch.Tensor):
            type = data.type()
            shape = list(data.size())
            shape[-2], shape[-1] = height, width
            data = torch.squeeze(data)
            pil_img = to_pil_image(data)
            pil_img = pil_img.resize((width, height))
            torch_img = pil_to_tensor(pil_img)
            if type == 'torch.FloatTensor':
                torch_img = (torch_img / 255).float()
            return torch.reshape(torch_img, shape)
       
        else:
            raise ValueError('Unknown data type to resize.')
    
    def segment_image(self, data):
        # Reformat input image
        data = data.numpy()
        data = np.squeeze(data * 255).astype(np.uint8)
        data = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)

        # Perform image segmentation
        segments = segment(data, self.sigma, self.seg_scale, self.min_size)
        pixels = segment_pixels(segments)
        return pixels

    def scale_values(self, values, method='normalize'):
        # Convert pytorch tensor to numpy array
        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()

        # Normalize set of values
        if method == 'normalize':
            values = values / np.sum(values)
            values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        
        # Compute zscores of set of values
        elif method == 'zscore':
            values = stats.zscore(values, axis=None)

        # Replace Nans with zeros
        values = np.nan_to_num(values, 0)
        return values
            

def plot_regions(input, comp, file=None, title=None):
    plt.figure(figsize=(8, 4))

    # Plot input data
    plt.subplot(1, 2, 1)
    data = np.swapaxes(np.swapaxes(input, 1, 2), 2, 3)
    if np.shape(data)[3] == 1:
        plt.imshow(data[0,:,:,0], cmap='binary')
    else:
        plt.imshow(data[0,:,:,:])
    plt.title('Input Image')

    # Plot competency gradients
    plt.subplot(1, 2, 2)
    plt.imshow(comp, cmap='coolwarm')
    plt.title('Regional Competency')

    # Add title
    if title is not None:
        plt.suptitle(title)

    # Display image and gradients
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()
    plt.close()