import string
import numpy as np

import torch
import torchvision
from torch.utils.data import ConcatDataset

from perception.datasets.custom_dataset import CustomDataset

def get_class_names(data):
    if data == 'emnist':
        return list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)
    elif data == 'lunar':
        return ['medium smooth', 'dark crater', 'dark bumpy', 'medium bumpy', 'sunny side',
                'light bumpy', 'light crater', 'light smooth'] + ['structures']
    elif data == 'speed':
        return ['speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 
                'speed limit 80', 'speed limit 100', 'speed limit 120'] + ['speed limit 20']
    elif data == 'pavilion':
        return ['facing building', 'facing parking', 'facing pavilion', 'hitting tree', 
                'middle area', 'open space 1', 'open space 2', 'wooded'] + ['pavilion']
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))

def get_num_classes(data):
    if data == 'emnist':
        return 10
    elif data == 'lunar':
        return 8
    elif data == 'speed':
        return 7
    elif data == 'pavilion':
        return 8
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))

def make_weights_for_balanced_classes(labels):
    total_lbls = len(labels)
    unique_lbls = np.unique(labels)
    weights = np.zeros(len(labels))
    for lbl in unique_lbls:
        count = len(np.where(labels.flatten() == lbl)[0])
        weights[labels.flatten() == lbl] = total_lbls / count                           
    return weights 

def setup_loader(data, batch_size=None, train=False, test=False, ood=False, example=False):
    if data == 'emnist':
        batch_size = 1000 if batch_size is None else batch_size
        if train:
            loader = torch.utils.data.DataLoader(
            torchvision.datasets.EMNIST('data/', train=True, download=True, split='digits',
                                transform=torchvision.transforms.Compose([
                                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                                    lambda img: torchvision.transforms.functional.hflip(img),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                batch_size=batch_size, shuffle=True)
            
        elif test and ood:
            loader = torch.utils.data.DataLoader(
            torchvision.datasets.EMNIST('data/', train=False, download=False, split='byclass',
                           transform=torchvision.transforms.Compose([
                             lambda img: torchvision.transforms.functional.rotate(img, -90),
                             lambda img: torchvision.transforms.functional.hflip(img),
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                         batch_size=batch_size, shuffle=False)

        elif test:
            loader = torch.utils.data.DataLoader(
            torchvision.datasets.EMNIST('data/', train=False, download=True, split='digits',
                           transform=torchvision.transforms.Compose([
                             lambda img: torchvision.transforms.functional.rotate(img, -90),
                             lambda img: torchvision.transforms.functional.hflip(img),
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                         batch_size=batch_size, shuffle=True)

    elif data == 'lunar':
        batch_size = 64 if batch_size is None else batch_size
        if train:
            train_dataset = CustomDataset('./data/lunar/', 'train')
            weights = make_weights_for_balanced_classes(train_dataset.labels)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    sampler=sampler)
            
        elif test and ood:
            loader = torch.utils.data.DataLoader(
            ConcatDataset([CustomDataset('./data/lunar/', 'ood'),
                           CustomDataset('./data/lunar/', 'test')]),
                       batch_size=batch_size, shuffle=False)
            
        elif test:
            loader = torch.utils.data.DataLoader(
                        CustomDataset('./data/lunar/', 'test'),
                        batch_size=batch_size, shuffle=True)
            
        elif ood:
            loader = torch.utils.data.DataLoader(
                        CustomDataset('./data/lunar/', 'ood'),
                        batch_size=batch_size, shuffle=False)
    
    elif data == 'speed':
        batch_size = 64 if batch_size is None else batch_size
        if train:
            train_dataset = CustomDataset('./data/speed/', 'train')
            weights = make_weights_for_balanced_classes(train_dataset.labels)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    sampler=sampler)
            
        elif test and ood:
            loader = torch.utils.data.DataLoader(
            ConcatDataset([CustomDataset('./data/speed/', 'ood'),
                           CustomDataset('./data/speed/', 'test')]),
                       batch_size=batch_size, shuffle=False)
            
        elif test:
            loader = torch.utils.data.DataLoader(
                        CustomDataset('./data/speed/', 'test'),
                        batch_size=batch_size, shuffle=True)
            
        elif ood:
            loader = torch.utils.data.DataLoader(
                        CustomDataset('./data/speed/', 'ood'),
                        batch_size=batch_size, shuffle=False)
    
    elif data == 'pavilion':
        batch_size = 64 if batch_size is None else batch_size
        if train:
            train_dataset = CustomDataset('./data/pavilion/', 'train')
            weights = make_weights_for_balanced_classes(train_dataset.labels)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    sampler=sampler)
            
        elif test and ood:
            loader = torch.utils.data.DataLoader(
            ConcatDataset([CustomDataset('./data/pavilion/', 'ood'),
                           CustomDataset('./data/pavilion/', 'test')]),
                       batch_size=batch_size, shuffle=False)
            
        elif test:
            loader = torch.utils.data.DataLoader(
                        CustomDataset('./data/pavilion/', 'test'),
                        batch_size=batch_size, shuffle=True)
            
        elif ood:
            loader = torch.utils.data.DataLoader(
                        CustomDataset('./data/pavilion/', 'ood'),
                        batch_size=batch_size, shuffle=False)
    
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))
    
    return loader