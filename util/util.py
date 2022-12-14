"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, normalized=True):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if image_numpy.shape[0] == 2:  # if two channels
            image_numpy = image_numpy[[0], ...]
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if image_numpy.shape[0] > 3:
            # only show the first 3 channel
            image_numpy = image_numpy[:3, ...]
        if normalized:
            factor = 255
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * factor  # post-processing: tranpose and scaling
            image_numpy = np.clip(image_numpy, a_min=0, a_max=factor)
        else:
            if np.max(image_numpy) > 1 + 1e-6:
                image_numpy = np.transpose(image_numpy, (1, 2, 0))
            else:
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    # print('image numpy shape:', image_numpy.shape)

    return image_numpy


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0, dtype='PIL'):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    if dtype == 'PIL':
        image_pil = Image.fromarray(image_numpy)
        h, w, _ = image_numpy.shape

        if aspect_ratio > 1.0:
            image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        if aspect_ratio < 1.0:
            image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
        image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def transform_image(img):
    return (img + 1.0) / 2.0

def transform_image_back(img):
    return (img - 0.5) * 2

def divide_after_transform(img1, img2, epsilon):
    img1_scale = transform_image(img1)
    img2_scale = transform_image(img2) + epsilon
    return img1_scale / img2_scale

# scale is derived with image of range [0, 1]
def multiply_after_transform(img1, scale):
    img1_scale = transform_image(img1)
    img1_scaled = img1_scale * scale
    img1_result = transform_image_back(img1_scaled)

    return img1_result
