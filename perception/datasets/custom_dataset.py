import os
import tqdm
import pickle
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from perception.regions.segment import segment, segment_pixels


class CustomDataset(Dataset):
    def __init__(self, data_dir, key, height=None, width=None):
        # Load saved data
        print('Loading {} dataset from {}...'.format(key, data_dir))
        file = os.path.join(data_dir, 'dataset.p')
        dataset = pickle.load(open(file, 'rb'))
        self.data = dataset[key]['data']
        self.labels = dataset[key]['labels']
        if height is not None and width is not None:
            self.data = self.resize_images(height, width)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:])
        return data, labels
    
    def resize_images(self, height, width):

        data = []
        for img in self.data:
            # resize image
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            pil_img = Image.fromarray(np.uint8(img * 255))
            pil_img = pil_img.resize((width, height))
            np_img = np.array(pil_img) / 255
            np_img = np.swapaxes(np.swapaxes(np_img, 1, 2), 0, 1)
            data.append(np_img)
            
        return np.stack(data)


class SegmentedDataset(Dataset):
    def __init__(self, data_dir, key, sigma, scale, min_size, height=None, width=None):
        # Load saved data
        print('Loading {} dataset from {}...'.format(key, data_dir))
        file = os.path.join(data_dir, 'dataset.p')
        dataset = pickle.load(open(file, 'rb'))
        self.data = dataset[key]['data']
        self.segmented_data, self.indices = self.segment_images(sigma, scale, min_size, height, width)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.segmented_data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.data[self.indices[idx],:,:,:])
        return data, labels
    
    def segment_images(self, sigma, scale, min_size, height=None, width=None):

        data, labels = [], []
        for idx, img in enumerate(self.data):
            # resize image
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            pil_img = Image.fromarray(np.uint8(img * 255))
            pil_img = pil_img.resize((width, height))
            np_img = np.array(pil_img)

            # segment image
            segments = segment(np_img, sigma, scale, min_size)
            all_pixels = segment_pixels(segments)

            # mask image
            # pixels = all_pixels[0]
            for pixels in all_pixels:
                masked_img = np_img.copy() / 255
                masked_img[pixels[0, :], pixels[1, :], :] = 1
                masked_img = np.swapaxes(np.swapaxes(masked_img, 1, 2), 0, 1)
                data.append(masked_img)
                labels.append(idx)

        return np.stack(data), np.stack(labels)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--scale', type=int, default=350)
    parser.add_argument('--min_size', type=int, default=60)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    args = parser.parse_args()

    if args.segment:
        train_dataset = SegmentedDataset(args.data_dir, 'train', args.sigma, args.scale, args.min_size, args.height, args.width)
        test_dataset  = SegmentedDataset(args.data_dir, 'test',  args.sigma, args.scale, args.min_size, args.height, args.width)
        ood_dataset   = SegmentedDataset(args.data_dir, 'ood',   args.sigma, args.scale, args.min_size, args.height, args.width)
        file = os.path.join(args.data_dir, 'segmented-dataset.p')
    else:
        train_dataset = CustomDataset(args.data_dir, 'train', args.height, args.width)
        test_dataset  = CustomDataset(args.data_dir, 'test',  args.height, args.width)
        ood_dataset   = CustomDataset(args.data_dir, 'ood',   args.height, args.width)
        file = os.path.join(args.data_dir, 'original-dataset.p')

    datasets = {'train': train_dataset, 'test': test_dataset, 'ood': ood_dataset}
    pickle.dump(datasets, open(file, 'wb'))

if __name__ == "__main__":
    main()