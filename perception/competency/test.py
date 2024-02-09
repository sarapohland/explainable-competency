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

def check_accuracy(pred, label):
    correct = (pred == label)
    return torch.sum(correct) / len(correct)

def plot_scores(id_data, ood_data, accuracy, score='alice', dataset='emnist'):
    plt.figure()
    plt.boxplot([id_data, ood_data])
    plt.xticks(np.arange(2)+1, ['ID', 'OOD'])
    plt.xlabel('dataset')
    plt.ylabel('{} score'.format(score))
    plt.title('{} scores of {} data: {}'.format(score, dataset, np.round(accuracy,2)))
    if not os.path.exists('results/{}/competency/'.format(dataset)):
        os.makedirs('results/{}/competency/'.format(dataset))
    # plt.show()
    plt.savefig('results/{}/competency/{}.png'.format(dataset, score))
    plt.close()

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default='data/')
    parser.add_argument('--model_dir', type=str, default='model/classify/')
    parser.add_argument('--thresh', type=float, default=None)
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

    # Check accuracy of competency decision
    if args.thresh is not None:
        estimator.set_threshold(args.thresh)
        pickle.dump(estimator, open(file, 'wb'))
    true = (all_labels < N)
    pred = estimator.comp_dec(all_scores, all_preds)
    acc  = check_accuracy(pred, true)

    # Separate in- and out-of-distribution data
    id_scores = all_scores[true]
    ood_scores = all_scores[[not x for x in true]]

    # Plot scores
    plot_scores(id_scores, ood_scores, acc, dataset=args.test_data)

if __name__ == "__main__":
    main()
