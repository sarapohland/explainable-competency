import os
import json
import torch
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from perception.network.model import NeuralNet
from perception.datasets.setup_dataloader import setup_loader

def get_accurary(true, pred):
    num_correct = torch.sum(pred == true)
    accuracy = num_correct / len(true)
    return accuracy.item()

def test(dataloader, model, device):
    model.eval()
    num_correct = 0
    trues, preds = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = torch.argmax(model(X), dim=1)
            true = torch.Tensor([int(yi.item()) for yi in y])
            trues.append(true)
            preds.append(pred)
    return torch.concatenate(trues), torch.concatenate(preds)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default='data/')
    parser.add_argument('--model_dir', type=str, default='model/')
    args = parser.parse_args()

    # Load trained model
    with open(os.path.join(args.model_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))

    # Create data loader
    test_loader = setup_loader(args.test_data, test=True)

    # Test model performance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    true, pred = test(test_loader, model, device)
    accuracy = get_accurary(true, pred)

    # Plot confusion matrix
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Accuracy: {}'.format(accuracy))
    plt.savefig(os.path.join(args.model_dir, 'confusion.png'))


if __name__ == "__main__":
    main()
