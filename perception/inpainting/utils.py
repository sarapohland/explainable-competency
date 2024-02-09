import torch
import numpy as np
from PIL import Image

def resize_image(imgs, height, width):
    data = []
    for img in imgs:
        # resize image
        np_img = img.numpy()
        np_img = np.swapaxes(np.swapaxes(np_img, 0, 1), 1, 2)
        pil_img = Image.fromarray(np.uint8(np_img * 255))
        pil_img = pil_img.resize((width, height))
        np_img = np.array(pil_img) / 255
        np_img = np.swapaxes(np.swapaxes(np_img, 1, 2), 0, 1)
        data.append(torch.from_numpy(np_img).float()) 
    return torch.stack(data)
