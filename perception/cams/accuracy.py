import os
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt

from PIL import Image
from scipy import stats
from tqdm import tqdm

from pytorch_grad_cam import (
    GradCAM, 
    GradCAMPlusPlus, 
    AblationCAM, 
    ScoreCAM, 
    EigenCAM, 
    LayerCAM
)
from captum.attr import (
    IntegratedGradients,
    NoiseTunnel,
    DeepLift,
    GuidedGradCam,
)

from perception.network.model import NeuralNet
from perception.datasets.setup_dataloader import setup_loader 
from perception.evaluation.accuracy import get_accuracy


def resize_image(data, height, width):
    shape = list(np.shape(data))
    shape[-2], shape[-1] = height, width
    data = np.squeeze(data)
    pil_img = Image.fromarray(data)
    pil_img = pil_img.resize((width, height))
    np_img = np.array(pil_img)
    return np.reshape(np_img, shape)

def scale_values(values, method='normalize'):
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
                
def main():
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--scale', type=str, default=None)
    parser.add_argument('--threshold', type=float, default='0.5')
    parser.add_argument('--output_dir', type=str, default='results/cam/')
    args = parser.parse_args()

    # Create folder to save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load trained classification model
    with open(args.model_dir + 'layers.json') as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(args.model_dir + 'model.pth'))
    target_layers = [model.net[0]]

    # Construct CAM objects
    grad_cam = GradCAM(model=model, target_layers=target_layers)
    grad_cam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers)
    ablation_cam = AblationCAM(model=model, target_layers=target_layers)
    score_cam = ScoreCAM(model=model, target_layers=target_layers)
    eigen_cam = EigenCAM(model=model, target_layers=target_layers)
    layer_cam = LayerCAM(model=model, target_layers=target_layers)
    cam_objects = [grad_cam, grad_cam_pp, ablation_cam, score_cam, eigen_cam, layer_cam]
    cam_names = ['grad_cam', 'grad_cam_pp', 'ablation_cam', 'score_cam', 'eigen_cam',  'layer_cam']

    # Construct Captum objects
    integrated_grads = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_grads)
    deep_lift = DeepLift(model)
    guided_grad_cam = GuidedGradCam(model, model.net[0])
    captum_objects = [integrated_grads, noise_tunnel, deep_lift, guided_grad_cam]
    captum_names = ['integrated_grads', 'smooth_grad', 'deep_lift', 'guided_grad_cam']

    # Load true segmentation labels
    label_dir = 'data/{}/'.format(args.test_data)
    segmentation = pickle.load(open(os.path.join(label_dir, 'segmentation.p'), 'rb'))
    seg_pixels = segmentation['pixels']
    seg_labels = segmentation['labels']

    # Get height and width of labeled images
    config = configparser.RawConfigParser()
    config.read(args.config_file)
    height = config.getint('size', 'height')
    width = config.getint('size', 'width')

    acc = {'grad_cam': [], 'guided_grad_cam': [], 'grad_cam_pp': [], 'integrated_grads': [], 'smooth_grad': [], 'deep_lift': [], 'score_cam': [], 'ablation_cam': [], 'eigen_cam': [], 'layer_cam': []}
    tpr = {'grad_cam': [], 'guided_grad_cam': [], 'grad_cam_pp': [], 'integrated_grads': [], 'smooth_grad': [], 'deep_lift': [], 'score_cam': [], 'ablation_cam': [], 'eigen_cam': [], 'layer_cam': []}
    tnr = {'grad_cam': [], 'guided_grad_cam': [], 'grad_cam_pp': [], 'integrated_grads': [], 'smooth_grad': [], 'deep_lift': [], 'score_cam': [], 'ablation_cam': [], 'eigen_cam': [], 'layer_cam': []}
    ppv = {'grad_cam': [], 'guided_grad_cam': [], 'grad_cam_pp': [], 'integrated_grads': [], 'smooth_grad': [], 'deep_lift': [], 'score_cam': [], 'ablation_cam': [], 'eigen_cam': [], 'layer_cam': []}
    npv = {'grad_cam': [], 'guided_grad_cam': [], 'grad_cam_pp': [], 'integrated_grads': [], 'smooth_grad': [], 'deep_lift': [], 'score_cam': [], 'ablation_cam': [], 'eigen_cam': [], 'layer_cam': []}
    
    # Get saliency maps for each sample in test set
    test_loader = setup_loader(args.test_data, batch_size=1, ood=True)
    for batch, (data, labels) in tqdm(enumerate(test_loader)):

        # Get true labels of image
        true_labels = seg_labels[batch]
        true_regions = np.zeros((height, width))
        for pixels, label in zip(seg_pixels[batch], true_labels):
            true_regions[pixels[0, :], pixels[1, :]] = label

        # Skip analysis if we do not have good labels
        if not np.any(true_regions == 1):
            continue

        # For all CAM objects...
        for cam_object, cam_name in zip(cam_objects, cam_names):
            # Get the class activation map
            cam_img = cam_object(input_tensor=data, targets=None)[0,:]
            
            # Resize and scale class activation map
            cam_img = scale_values(cam_img, args.scale)
            cam_img = resize_image(cam_img, height, width)
            
            # Highlight regions above threshold
            cam_img_thresh = np.zeros_like(cam_img)
            cam_img_thresh[cam_img > args.threshold] = 1

            # Compute accuracy
            accuracy = get_accuracy(true_regions, cam_img_thresh)
            acc[cam_name].append(accuracy['ACC'])
            tpr[cam_name].append(accuracy['TPR'])
            tnr[cam_name].append(accuracy['TNR'])
            ppv[cam_name].append(accuracy['PPV'])
            npv[cam_name].append(accuracy['NPV'])

        # For all Captum objects...
        baseline = torch.zeros_like(data)
        outputs  = model(data)
        _, preds = torch.max(outputs, 1)
        for captum_object, captum_name in zip(captum_objects, captum_names):
            # Get the class activation map
            if captum_name == 'integrated_grads' or captum_name == 'deep_lift':
                attributions = captum_object.attribute(data, baselines=baseline, target=preds)
            elif captum_name == 'smooth_grad':
                attributions = captum_object.attribute(data, nt_type='smoothgrad', baselines=baseline, target=preds)
            elif captum_name == 'guided_grad_cam':
                attributions = captum_object.attribute(data, target=preds)
            else:
                attributions = captum_object.attribute(data)
            cam_img = torch.mean(attributions[0,:], dim=0).detach().numpy()

            # Resize class activation map
            cam_img = scale_values(cam_img, args.scale)
            cam_img = resize_image(cam_img, height, width)
            
            # Highlight regions above threshold
            cam_img_thresh = np.zeros_like(cam_img)
            cam_img_thresh[cam_img > args.threshold] = 1

            # Compute accuracy
            accuracy = get_accuracy(true_regions, cam_img_thresh)
            acc[captum_name].append(accuracy['ACC'])
            tpr[captum_name].append(accuracy['TPR'])
            tnr[captum_name].append(accuracy['TNR'])
            ppv[captum_name].append(accuracy['PPV'])
            npv[captum_name].append(accuracy['NPV'])

    # Plot accuracy of estimates
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame.from_dict(acc)
    methods = cam_names + captum_names
    bp = df.boxplot(vert=False, column=methods)
    plt.title('Accuracy of Unfamilarity Predictions')
    plt.xlabel('Method Accuracy')
    plt.savefig(os.path.join(args.output_dir, 'accuracy.png'), bbox_inches='tight')

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