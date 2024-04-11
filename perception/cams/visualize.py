import os
import json
import torch
import argparse
import matplotlib.pyplot as plt

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

def main():
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/cam/figures/')
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

    # Get saliency maps for each sample in test set
    test_loader = setup_loader(args.test_data, batch_size=1, ood=True)
    for batch, (data, labels) in enumerate(test_loader):
        for cam_object, cam_name in zip(cam_objects, cam_names):
            cam_img = cam_object(input_tensor=data, targets=None)
            plt.imshow(cam_img[0,:], cmap='coolwarm')
            img_name = '{}_{}.png'.format(batch, cam_name)
            plt.savefig(os.path.join(args.output_dir, img_name))
            plt.close()

        baseline = torch.zeros_like(data)
        outputs  = model(data)
        _, preds = torch.max(outputs, 1)
        for captum_object, captum_name in zip(captum_objects, captum_names):
            if captum_name == 'integrated_grads' or captum_name == 'deep_lift':
                attributions = captum_object.attribute(data, baselines=baseline, target=preds)
            elif captum_name == 'smooth_grad':
                attributions = captum_object.attribute(data, nt_type='smoothgrad', baselines=baseline, target=preds)
            elif captum_name == 'guided_grad_cam':
                attributions = captum_object.attribute(data, target=preds)
            else:
                attributions = captum_object.attribute(data)
            plt.imshow(torch.mean(attributions[0,:], dim=0).detach().numpy(), cmap='coolwarm')
            img_name = '{}_{}.png'.format(batch, captum_name)
            plt.savefig(os.path.join(args.output_dir, img_name))
            plt.close()

if __name__=="__main__":
    main()