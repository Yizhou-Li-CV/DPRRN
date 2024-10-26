import cv2
import numpy as np
from pylab import plt

import os
from skimage import transform
from scipy import ndimage

import argparse


def read_files(path):
    file_names = os.listdir(path)
    file_indexs = list(set([file_name.replace('_left.pgm', '').replace('_right.pgm', '') for file_name in file_names]))
    file_indexs = sorted(file_indexs)
    for idx in file_indexs:
        left_img_name = os.path.join(path, idx + '_left.pgm')
        right_img_name = os.path.join(path, idx + '_right.pgm')
        yield left_img_name, right_img_name


def _crop_image_central_fov(images, crop_y, crop_x):
    """ Crop images
    Args:
      images: [..., H, W, C] #images, height, width, #channels.
    Returns:
      [..., #rows * P, #cols * P, C] cropped images
    """
    offset_y = (images.shape[-3] - crop_y) // 2
    offset_x = (images.shape[-2] - crop_x) // 2

    return images[..., offset_y:offset_y + crop_y, offset_x:offset_x + crop_x, :]


def process_dp_raw(filename, center_crop=True, resize=True):
    img = cv2.imread(filename, -1) - 4096
    print(img.shape, type(img), np.min(img), np.max(img))
    img = img / 65535
    print(img.shape, type(img), np.min(img), np.max(img))
    new_img = np.zeros((756, 2016, 3))
    new_img[..., 0] = img
    new_img[..., 1] = img
    new_img[..., 2] = img

    new_img = cv2.resize(new_img, (2016, 1512), interpolation=cv2.INTER_LINEAR)

    if center_crop:
        new_img = _crop_image_central_fov(new_img, 1008, 1344)

    # # real-world data needs to be cropped 1008 * 1344 if using kernel from ICCV2021 calibrated ones
    # # and, resize scale must be determined by original size, because here sensor width also changed
    if resize:
        scale_factor = 2016 / 800
        target_w = int(1344 / scale_factor)
        target_h = int(1008 / scale_factor)
        target_h = target_h // 8 * 8
        target_w = target_w // 8 * 8

        print(target_h, target_w)

        new_img = cv2.resize(new_img, dsize=(target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return new_img


def save_L_R_C(save_path, left_img, right_img, center_img, idx, resize_shape=None):
    left_img_path = os.path.join(save_path, f'{idx}_L.png')
    right_img_path = os.path.join(save_path, f'{idx}_R.png')
    C_img_path = os.path.join(save_path, f'{idx}_C.png')

    if resize_shape is not None:
        left_img = transform.resize(left_img, output_shape=resize_shape)
        right_img = transform.resize(right_img, output_shape=resize_shape)
        center_img = transform.resize(center_img, output_shape=resize_shape)

    left_img = np.clip(left_img, a_min=0, a_max=1)
    right_img = np.clip(right_img, a_min=0, a_max=1)
    center_img = np.clip(center_img, a_min=0, a_max=1)

    plt.imsave(left_img_path, left_img)
    plt.imsave(right_img_path, right_img)
    plt.imsave(C_img_path, center_img)


def filter_pepper_noise(img, thresh=0.0025):
    img_filtered = ndimage.median_filter(img, size=5)
    noise_mask = np.logical_or(np.abs(img - 0.75) <= thresh, np.abs(img - 0.25) <= thresh)
    noise_mask = np.logical_or(noise_mask, np.abs(img - 1) <= thresh)
    img[noise_mask] = img_filtered[noise_mask]
    return img

def main(raw_dir, transfered_dir):
    gamma = 2.2
    color_scale = 1

    save_path = os.path.join(transfered_dir, 'dp_rainy')
    save_path_gt = os.path.join(transfered_dir, 'dp_gt')

    if not os.path.exists(transfered_dir):
        os.makedirs(transfered_dir)
        os.makedirs(save_path)
        os.makedirs(save_path_gt)

    counter = 0

    for i, [left_img_name, right_img_name] in enumerate(read_files(raw_dir)):

        counter += 1

        print('Processing', left_img_name)
        left_img = process_dp_raw(left_img_name, side='left')
        right_img = process_dp_raw(right_img_name, side='right')

        left_img = filter_pepper_noise(left_img, thresh=0.003)
        right_img = filter_pepper_noise(right_img, thresh=0.003)

        left_img *= color_scale
        right_img *= color_scale

        left_img = np.clip(left_img, a_min=0, a_max=1)
        right_img = np.clip(right_img, a_min=0, a_max=1)

        left_img_gamma = np.float_power(left_img, 1 / gamma)
        right_img_gamma = np.float_power(right_img, 1 / gamma)

        center_img_gamma = (left_img_gamma + right_img_gamma) / 2
        center_img_gamma = np.clip(center_img_gamma, a_min=0, a_max=1)

        save_L_R_C(save_path, left_img_gamma, right_img_gamma, center_img_gamma, counter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--transferred_dir', type=str, required=True)
    args = parser.parse_args()

    main(args.raw_dir, args.transferred_dir)
