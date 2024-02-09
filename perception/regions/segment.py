import time
import torch
import numpy as np

from skimage.segmentation import felzenszwalb


def segment_img(input, segments):
    colors = []
    for segment in np.unique(segments):
        colors.append(np.random.randint(0, 255, size=(3,), dtype=int))
    
    output = np.zeros_like(input)
    height, width, _ = np.shape(input)
    for i in range(height):
        for j in range(width):   
            output[i, j, :] = colors[segments[i, j]]
    return output

def segment_pixels(segments):
    all_pixels = []
    for segment in np.unique(segments):
        # Get pixels coorresponding to current segment
        pixels = np.array(np.where(segments == segment))
        all_pixels.append(torch.LongTensor(pixels))
    return all_pixels

def segment(img, sigma, scale, min_size):
    return felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
   
def main():
    import os
    import argparse
    from PIL import Image
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--img_folder', type=str, default='data/')
    parser.add_argument('--output_dir', type=str, default='images/')
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--scale', type=int, default=750)
    parser.add_argument('--min_size', type=int, default=50)
    args = parser.parse_args()

    # Create directory to store segmented images
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _, cluster_names, _ = list(os.walk(args.img_folder))[0]
    for idx, cluster_name in enumerate(cluster_names):
        cluster_dir = os.path.join(args.img_folder, cluster_name)
        for file in os.listdir(cluster_dir):
            if not (file.endswith('.jpg') or file.endswith('.png')):
                continue

            # Read input image
            img = Image.open(os.path.join(cluster_dir, file))
            height = args.height if args.height is not None else img.size[1] 
            width  = args.width  if args.width  is not None else img.size[0] 
            img = img.resize((width, height))
            img = np.array(img)

            # Segment image
            segments = segment(img, args.sigma, args.scale, args.min_size)
            output = segment_img(img, segments)

            # Plot segmented image
            plt.subplot(1, 2, 1)
            im = plt.imshow(img)
            plt.subplot(1, 2, 2)
            im = plt.imshow(output)
            plt.savefig(os.path.join(args.output_dir, file))

if __name__=="__main__":
    main()