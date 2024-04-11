import os
import json
import torch
import pickle
import string
import argparse
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from perception.network.model import NeuralNet
from perception.datasets.setup_dataloader import setup_loader, get_num_classes

torch.manual_seed(0)

def plot_scores(correct_data, incorrect_data, ood_data, dataset='emnist'):
    plt.figure()
    plt.boxplot([correct_data, incorrect_data, ood_data])
    plt.xticks(np.arange(3)+1, ['ID-Correct', 'ID-Incorrect', 'OOD'])
    plt.ylabel('Competency Score')
    plt.title('Competency Scores of {} Data'.format(dataset.capitalize()))
    if not os.path.exists('results/{}/competency/'.format(dataset)):
        os.makedirs('results/{}/competency/'.format(dataset))
    # plt.show()
    plt.savefig('results/{}/competency/scores.png'.format(dataset))
    plt.close()

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default='data/')
    parser.add_argument('--model_dir', type=str, default='model/classify/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Load trained model
    with open(args.model_dir + 'layers.json') as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(args.model_dir + 'model.pth'))
    model.eval()

    # Create data loader
    N = get_num_classes(args.test_data)
    test_loader = setup_loader(args.test_data, test=True, ood=True)

    # Load pretrained competency estimator
    file = os.path.join(args.model_dir, 'alice.p')
    estimator = pickle.load(open(file, 'rb'))

    # Estimate competency scores for test data
    all_labels, all_preds, all_scores  = [], [], []
    for inputs, labels in tqdm(test_loader):
        with torch.set_grad_enabled(False):
            # Get predictions of trained model
            outputs  = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Compute competency scores
            scores = estimator.comp_score(inputs, outputs)

            # Visualize image and competency score
            if args.debug:
                for data, comp in zip(inputs, scores):
                    plt.imshow(np.swapaxes(np.swapaxes(data, 0, 1), 1, 2))
                    plt.title('Competency Score: {}'.format(comp))
                    plt.show()

            # Collect scores and true labels
            all_preds.append(preds)
            all_labels.append(labels.flatten())
            all_scores.append(scores)

    # Convert collected data to tensors
    all_preds  = torch.hstack(all_preds)
    all_labels = torch.hstack(all_labels)
    all_scores = torch.hstack(all_scores)

    # Separate in- and out-of-distribution data
    id_idx = (all_labels < N)
    id_preds = all_preds[id_idx]
    id_labels = all_labels[id_idx]
    id_scores = all_scores[id_idx]
    ood_scores = all_scores[[not x for x in id_idx]]

    # Separate correctly classified and misclassified ID data
    correct_idx = (id_preds == id_labels)
    correct_scores = id_scores[correct_idx]
    incorrect_scores = id_scores[[not x for x in correct_idx]]

    # Plot scores
    plot_scores(correct_scores, incorrect_scores, ood_scores, dataset=args.test_data)

if __name__ == "__main__":
    main()
