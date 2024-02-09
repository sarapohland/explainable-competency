import os
import json
import torch
import pickle
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from perception.network.model import NeuralNet
from perception.inpainting.autoencoder import AutoEncoder
from perception.datasets.setup_dataloader import setup_loader

from perception.regions.segment import segment, segment_img
from perception.regions.cropping import Cropping
from perception.regions.masking import Masking
from perception.regions.perturbation import Perturbation
from perception.regions.gradients import Gradients
from perception.regions.reconstruction import Reconstruction

if __name__=="__main__":

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--autoencoder_dir', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results/figures/')
    args = parser.parse_args()

    # Create folder to save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    subfolders = [os.path.join(args.output_dir, folder) for folder in ['cropping', 'masking', 'perturbation', 'gradients', 'reconstruction']]
    for folder in subfolders:
        if not os.path.exists(folder):
            os.makedirs(folder)

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

    # Visualize single data sample
    if args.sample is not None:
        for batch, (data, label) in enumerate(test_loader):
            if batch < args.sample:
                continue

            # Display original image
            plt.imshow(np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)[0,:,:,:])
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(args.output_dir, 'image.png'))
            plt.close()

            #################
            ## 1) CROPPING ##
            #################

            # Display image partitioning
            plt.imshow(np.swapaxes(np.swapaxes(data, 1, 2), 2, 3)[0,:,:,:])
            ax = plt.gca()
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=cropping.cell_width))
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=cropping.cell_height))
            ax.grid(which='major', axis='both', linestyle='-')
            ax.autoscale_view('tight')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.savefig(os.path.join(subfolders[0], 'partitioning.png'))
            plt.close()

            # Partition image into cells and crop
            idx = 0
            img_height, img_width = data.size()[2], data.size()[3]
            for start_x in range(0, img_height-1, cropping.cell_height):
                for start_y in range(0, img_width-1, cropping.cell_width):
                    # Crop image
                    end_x = (start_x + cropping.cell_height) if (start_x + cropping.cell_height) < img_height else img_height
                    end_y = (start_y + cropping.cell_width)  if (start_y + cropping.cell_width)  < img_width  else img_width
                    cropped_img = data[:, :, start_x:end_x, start_y:end_y].clone()

                    # Plot cropped image
                    plt.imshow(np.swapaxes(np.swapaxes(cropped_img, 1, 2), 2, 3)[0,:,:,:])
                    ax = plt.gca()
                    ax.autoscale_view('tight')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(os.path.join(subfolders[0], 'cropped_{}.png'.format(idx)))
                    plt.close()
                    idx += 1

            # Plot incompetency dependency
            crop_regions = cropping.get_regions(data)
            plt.imshow(crop_regions, cmap='coolwarm')
            # ax = plt.gca()
            # ax.xaxis.set_major_locator(plticker.MultipleLocator(base=cropping.cell_width/2))
            # ax.yaxis.set_major_locator(plticker.MultipleLocator(base=cropping.cell_height/2))
            # ax.grid(which='major', axis='both', linestyle='-')
            # ax.autoscale_view('tight')
            # plt.xticks(color='w')
            # plt.yticks(color='w')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[0], 'dependency.png'))
            plt.close()
            
            ################
            ## 2) MASKING ##
            ################

            # Segment image
            resized = masking.resize_image(data)
            all_pixels = masking.segment_image(resized)
   
            # Display segmented image
            resized_disp = resized.numpy()
            resized_disp = np.squeeze(resized_disp * 255).astype(np.uint8)
            resized_disp = np.swapaxes(np.swapaxes(resized_disp, 0, 1), 1, 2)
            segments = segment(resized_disp, masking.sigma, masking.scale, masking.min_size)
            disp_img = segment_img(resized_disp, segments)
            plt.imshow(disp_img)
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[1], 'segmentation.png'))
            plt.close()

            for idx, pixels in enumerate(all_pixels):
                # Create a masked image
                masked_img = masking.mask_image(resized).clone()
                masked_img[:, :, pixels[0, :], pixels[1, :]] = resized[:, :, pixels[0, :], pixels[1, :]].clone()

                # Plot masked image
                plt.imshow(np.swapaxes(np.swapaxes(masked_img, 1, 2), 2, 3)[0,:,:,:])
                ax = plt.gca()
                ax.autoscale_view('tight')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(subfolders[1], 'masked_{}.png'.format(idx)))
                plt.close()

            # Plot incompetency dependency
            mask_regions = masking.get_regions(data)
            plt.imshow(mask_regions, cmap='coolwarm')
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[1], 'dependency.png'))
            plt.close()
            
            #####################
            ## 3) PERTURBATION ##
            #####################

            # Segment image
            resized = perturb.resize_image(data)
            all_pixels = perturb.segment_image(resized)
   
            # Display segmented image
            resized_disp = resized.numpy()
            resized_disp = np.squeeze(resized_disp * 255).astype(np.uint8)
            resized_disp = np.swapaxes(np.swapaxes(resized_disp, 0, 1), 1, 2)
            segments = segment(resized_disp, perturb.sigma, perturb.scale, perturb.min_size)
            disp_img = segment_img(resized_disp, segments)
            plt.imshow(disp_img)
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[2], 'segmentation.png'))
            plt.close()

            for idx, pixels in enumerate(all_pixels):
                # Perturb segment of image
                masked_img = perturb.mask_image(resized).clone()
                perturbed_img = resized.clone()
                perturbed_img[:, :, pixels[0, :], pixels[1, :]] = masked_img[:, :, pixels[0, :], pixels[1, :]].clone()

                # Plot perturbed image
                plt.imshow(np.swapaxes(np.swapaxes(perturbed_img, 1, 2), 2, 3)[0,:,:,:])
                ax = plt.gca()
                ax.autoscale_view('tight')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(subfolders[2], 'perturbed_{}.png'.format(idx)))
                plt.close()

            # Plot incompetency dependency
            perturb.compute_competency(data)
            pert_regions = perturb.get_regions(data)
            plt.imshow(pert_regions, cmap='coolwarm')
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[2], 'dependency.png'))
            plt.close()
            
            ##################
            ## 4) GRADIENTS ##
            ##################

            # Compute competency gradients
            gradients.comp = None
            gradients.gradients = []
            gradients.activations = []
            gradients.compute_competency(data)
            comp_grads = gradients.get_heatmap(gradients.comp)

            # Display gradients before segmentation
            plt.imshow(comp_grads[0].detach(), cmap='coolwarm')
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[3], 'gradients.png'))
            plt.close()
            
            # Plot incompetency dependency
            gradients.comp = None
            gradients.gradients = []
            gradients.activations = []
            gradients.compute_competency(data)
            grad_regions = gradients.get_regions(data)
            plt.imshow(grad_regions, cmap='coolwarm')
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[3], 'dependency.png'))
            plt.close()
            
            #######################
            ## 5) RECONSTRUCTION ##
            #######################

            # Segment image
            resized = reconstr.resize_image(data)
            all_pixels = reconstr.segment_image(resized)
   
            # Display segmented image
            resized_disp = resized.numpy()
            resized_disp = np.squeeze(resized_disp * 255).astype(np.uint8)
            resized_disp = np.swapaxes(np.swapaxes(resized_disp, 0, 1), 1, 2)
            segments = segment(resized_disp, reconstr.sigma, reconstr.scale, reconstr.min_size)
            disp_img = segment_img(resized_disp, segments)
            plt.imshow(disp_img)
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[4], 'segmentation.png'))
            plt.close()

            masked_imgs = []
            for pixels in all_pixels:
                # Create a mask tensor for the current segment
                masked_img = resized.clone()
                masked_img[:, :, pixels[0, :], pixels[1, :]] = 1
                masked_imgs.append(masked_img)

            # Reconstruct missing segments of image
            masked_imgs = torch.vstack(masked_imgs)
            pred_imgs = reconstr.const(masked_imgs)
            reconst_imgs = resized.expand(pred_imgs.size()).clone()
            for i, pixels in enumerate(all_pixels):
                reconst_imgs[i, :, pixels[0, :], pixels[1, :]] = pred_imgs[i, :, pixels[0, :], pixels[1, :]].clone()

            # Plot masked and reconstructed images
            for idx, (masked_img, reconst_img) in enumerate(zip(masked_imgs, reconst_imgs)):
                plt.imshow(np.swapaxes(np.swapaxes(masked_img, 0, 1), 1, 2))
                ax = plt.gca()
                ax.autoscale_view('tight')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(subfolders[4], 'masked_{}.png'.format(idx)))
                plt.close()
                
                plt.imshow(np.swapaxes(np.swapaxes(reconst_img.detach(), 0, 1), 1, 2))
                ax = plt.gca()
                ax.autoscale_view('tight')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(subfolders[4], 'reconstructed_{}.png'.format(idx)))
                plt.close()
            
            # Plot incompetency dependency
            reco_regions = reconstr.get_regions(data)
            plt.imshow(reco_regions, cmap='coolwarm')
            ax = plt.gca()
            ax.autoscale_view('tight')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(subfolders[4], 'dependency.png'))
            plt.close()
                
            break
