import os
import time
import json
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from perception.network.model import NeuralNet
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.setup_dataloader import setup_loader

from perception.regions.cropping import Cropping
from perception.regions.masking import Masking
from perception.regions.perturbation import Perturbation
from perception.regions.gradients import Gradients
from perception.regions.reconstruction import Reconstruction


def get_accuracy(true, pred):
    true = true.astype(int).flatten()
    pred = pred.astype(int).flatten()
    cm = confusion_matrix(true, pred, labels=[0,1])

    FP = cm[0,1]  
    FN = cm[1,0]
    TP = cm[1,1]
    TN = cm[0,0]

    TPR = TP/(TP+FN) # true positive rate/sensitivity/recall
    TNR = TN/(TN+FP) # true negative rate/specificity
    PPV = TP/(TP+FP) # positive predictive value/precision
    NPV = TN/(TN+FN) # negative predictive value
    FPR = FP/(FP+TN) # false positive rate/fall out
    FNR = FN/(TP+FN) # false negative rate
    FDR = FP/(TP+FP) # false discovery rate
    ACC = (TP+TN)/(TP+FP+FN+TN) # overall accuracy

    accuracies = {'ACC': ACC, 'TPR': TPR, 'TNR': TNR, 'FPR': FPR, \
                  'FNR': FNR, 'PPV': PPV, 'NPV': NPV, 'FDR': FDR}
    return accuracies

def main():
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--autoencoder_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Create folder to save results
    output_dir = 'results/{}/'.format(args.test_data)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Instantiate all of the regional competency approaches
    cropping  = Cropping(args.config_file, model, alice)
    masking   = Masking(args.config_file, model, alice)
    perturb   = Perturbation(args.config_file, model, alice)
    gradients = Gradients(args.config_file, model, alice)
    reconstr  = Reconstruction(args.config_file, model, alice, constructor)

    # Load true segmentation labels
    label_dir = 'data/{}/'.format(args.test_data)
    segmentation = pickle.load(open(os.path.join(label_dir, 'segmentation.p'), 'rb'))
    seg_pixels = segmentation['pixels']
    seg_labels = segmentation['labels']

    acc = {'Cropping': [], 'Segmentation': [], 'Perturbation': [], 'Gradients': [], 'Reconstruction': []}
    tpr = {'Cropping': [], 'Segmentation': [], 'Perturbation': [], 'Gradients': [], 'Reconstruction': []}
    tnr = {'Cropping': [], 'Segmentation': [], 'Perturbation': [], 'Gradients': [], 'Reconstruction': []}
    ppv = {'Cropping': [], 'Segmentation': [], 'Perturbation': [], 'Gradients': [], 'Reconstruction': []}
    npv = {'Cropping': [], 'Segmentation': [], 'Perturbation': [], 'Gradients': [], 'Reconstruction': []}
    
    for batch, (data, _) in enumerate(test_loader):

        # Get true labels of image
        true_labels = seg_labels[batch]
        true_regions = np.zeros((cropping.height, cropping.width))
        for pixels, label in zip(seg_pixels[batch], true_labels):
            true_regions[pixels[0, :], pixels[1, :]] = label

        # Skip analysis if we do not have good labels
        if not np.any(true_regions == 1):
            continue

        # Compute competency prediction
        gradients.comp = None
        gradients.gradients = []
        gradients.activations = []
        comp = gradients.compute_competency(data)
        cropping.comp = masking.comp = perturb.comp = reconstr.comp = comp
        cropping.regions = masking.regions = perturb.regions = gradients.regions = reconstr.regions = None

        # # Skip analysis if competency is high
        # if comp > 0.95:
        #     continue

        # Get regional competency estimates
        grad_regions = gradients.get_regions(data)
        crop_regions = cropping.get_regions(data)
        mask_regions = masking.get_regions(data)
        pert_regions = perturb.get_regions(data)
        reco_regions = reconstr.get_regions(data)

        # Highlight regions above threshold
        crop_regions_thresh = np.zeros_like(crop_regions)
        crop_regions_thresh[crop_regions > cropping.thresh] = 1
        mask_regions_thresh = np.zeros_like(mask_regions)
        mask_regions_thresh[mask_regions > masking.thresh] = 1
        pert_regions_thresh = np.zeros_like(pert_regions)
        pert_regions_thresh[pert_regions > perturb.thresh] = 1
        grad_regions_thresh = np.zeros_like(grad_regions)
        grad_regions_thresh[grad_regions > gradients.thresh] = 1
        reco_regions_thresh = np.zeros_like(reco_regions)
        reco_regions_thresh[reco_regions > reconstr.thresh] = 1

        # Compute accuracy
        crop_accuracy = get_accuracy(true_regions, crop_regions_thresh)
        acc['Cropping'].append(crop_accuracy['ACC'])
        tpr['Cropping'].append(crop_accuracy['TPR'])
        tnr['Cropping'].append(crop_accuracy['TNR'])
        ppv['Cropping'].append(crop_accuracy['PPV'])
        npv['Cropping'].append(crop_accuracy['NPV'])

        mask_accuracy = get_accuracy(true_regions, mask_regions_thresh)
        acc['Segmentation'].append(mask_accuracy['ACC'])
        tpr['Segmentation'].append(mask_accuracy['TPR'])
        tnr['Segmentation'].append(mask_accuracy['TNR'])
        ppv['Segmentation'].append(mask_accuracy['PPV'])
        npv['Segmentation'].append(mask_accuracy['NPV'])

        pert_accuracy = get_accuracy(true_regions, pert_regions_thresh)
        acc['Perturbation'].append(pert_accuracy['ACC'])
        tpr['Perturbation'].append(pert_accuracy['TPR'])
        tnr['Perturbation'].append(pert_accuracy['TNR'])
        ppv['Perturbation'].append(pert_accuracy['PPV'])
        npv['Perturbation'].append(pert_accuracy['NPV'])

        grad_accuracy = get_accuracy(true_regions, grad_regions_thresh)
        acc['Gradients'].append(grad_accuracy['ACC'])
        tpr['Gradients'].append(grad_accuracy['TPR'])
        tnr['Gradients'].append(grad_accuracy['TNR'])
        ppv['Gradients'].append(grad_accuracy['PPV'])
        npv['Gradients'].append(grad_accuracy['NPV'])

        reco_accuracy = get_accuracy(true_regions, reco_regions_thresh)
        acc['Reconstruction'].append(reco_accuracy['ACC'])
        tpr['Reconstruction'].append(reco_accuracy['TPR'])
        tnr['Reconstruction'].append(reco_accuracy['TNR'])
        ppv['Reconstruction'].append(reco_accuracy['PPV'])
        npv['Reconstruction'].append(reco_accuracy['NPV'])

        if args.debug:
            show_image = True
            true_regions[true_regions == -1] = 0.5
            fig = plt.figure(figsize=(12, 8))

            if show_image:
                plt.subplot(2, 3, 1)
                data = np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)
                im = plt.imshow(data[0,:,:,:])
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(2, 3, 2)
                im = plt.imshow(true_regions, cmap='coolwarm', vmin=0, vmax=1)
                plt.title('True Segmentation Labels')
                plt.axis('off')
            
            else:
                plt.subplot(2, 3, 1)
                im = plt.imshow(true_regions, cmap='coolwarm', vmin=0, vmax=1)
                plt.title('True Segmentation Labels')
                plt.axis('off')

                plt.subplot(2, 3, 2)
                im = plt.imshow(crop_regions, cmap='coolwarm', vmin=0, vmax=1)
                plt.title('Cropping Accuracy: {}'.format(round(acc['Cropping'][-1].item(), 3)))
                plt.axis('off')

            plt.subplot(2, 3, 3)
            im = plt.imshow(mask_regions_thresh, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Segmentation Accuracy: {}'.format(round(acc['Segmentation'][-1].item(), 3)))
            plt.axis('off')

            plt.subplot(2, 3, 4)
            im = plt.imshow(pert_regions_thresh, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Perturbation Accuracy: {}'.format(round(acc['Perturbation'][-1].item(), 3)))
            plt.axis('off')

            plt.subplot(2, 3, 5)
            im = plt.imshow(grad_regions_thresh, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Gradients Accuracy: {}'.format(round(acc['Gradients'][-1].item(), 3)))
            plt.axis('off')

            plt.subplot(2, 3, 6)
            im = plt.imshow(reco_regions_thresh, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Reconstruction Accuracy: {}'.format(round(acc['Reconstruction'][-1].item(), 3)))
            plt.axis('off')

            plt.tight_layout()
            # plt.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # plt.colorbar(im, cax=cbar_ax)
            plt.suptitle('Competency Score: {}'.format(round(comp.item(), 3)), size='x-large')
            plt.show()

    # Plot accuracy of estimates
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame.from_dict(acc)
    methods = ['Reconstruction', 'Gradients', 'Perturbation', 'Segmentation', 'Cropping']
    bp = df.boxplot(vert=False, column=methods)
    plt.title('Accuracy of Unfamilarity Predictions')
    plt.xlabel('Method Accuracy')
    plt.savefig(os.path.join(output_dir, 'accuracy.png'), bbox_inches='tight')

    # Print average accuracies
    for method in methods:
        print(method)
        print('Avg ACC: {}'.format(np.nanmean(acc[method])))
        print('Avg TPR: {}'.format(np.nanmean(tpr[method])))
        print('Avg TNR: {}'.format(np.nanmean(tnr[method])))
        print('Avg PPV: {}'.format(np.nanmean(ppv[method])))
        print('Avg NPV: {}'.format(np.nanmean(npv[method])))

if __name__=="__main__":
    main()