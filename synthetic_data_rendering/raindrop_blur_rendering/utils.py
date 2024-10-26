import numpy as np
import os
import cv2

import math

def check_dir(dir_name):
    '''check directory existence, if not, create it'''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_dist_with_angle(D, fov_angle, theta_angle):
    theta = theta_angle / 180 * np.pi
    fov = fov_angle / 2 / 180 * np.pi
    top_size_D = D / (1 + math.tan(fov) / math.tan(theta))
    bottom_side_D = D / (1 - math.tan(fov) / math.tan(theta))

    return top_size_D, bottom_side_D

def alpha_blending(imgA, imgB, alpha_mask):
    '''Alpha blending two images using given alpha mask'''
    imgA = np.asarray(imgA, dtype=np.float)
    imgB = np.asarray(imgB, dtype=np.float)
    imgOut = imgA + imgB * (1 - alpha_mask)
    imgOut = np.clip(np.round(imgOut), a_min=0, a_max=255)
    imgOut = np.asarray(imgOut, dtype=np.uint8)
    return imgOut

def gray_to_rgb(img):
    return np.stack([img, img, img], axis=-1)

def split_scenes(scene_num, train_ratio):
    '''Split scenes into train and test sets'''
    scenes = np.array(list(range(scene_num)))
    np.random.seed(66)
    np.random.shuffle(scenes)
    train_scene_num = int(scene_num * train_ratio)
    train_scene_indexes = scenes[:train_scene_num]
    test_scene_indexes = scenes[train_scene_num:]

    train_image_indexes = [idx * 4 + j for idx in train_scene_indexes for j in range(4)]
    test_image_indexes = [idx * 4 + j for idx in test_scene_indexes for j in range(4)]

    return train_image_indexes, test_image_indexes

def load_aif_file_info(dir_name):
    '''Load file info from AIF directory using the file name pattern'''
    file_list = os.listdir(f'{dir_name}/rainy/')
    file_info = {}
    for file_path in file_list:
        idx, angle, _, _, depth, _ = file_path.split('_')
        idx = int(idx)
        if idx not in file_info:
            file_info[idx] = {
                'angle': float(angle),
                'depth': float(depth),
                'file_name_L': file_path if 'L' in file_path else None
            }
    return file_info

def load_images(dir_name, file_name_L, file_name_R):
    '''Load AiF images'''
    images_rgb_L = f'{dir_name}/rainy/{file_name_L}'
    images_gt_L = f'{dir_name}/clean/{file_name_L}'
    images_gt_R = f'{dir_name}/clean/{file_name_R}'

    if not os.path.exists(images_rgb_L) or not os.path.exists(images_gt_L):
        print('File not exist', images_rgb_L, images_gt_L)
        return None

    img_rgb_L = cv2.imread(images_rgb_L, 1).astype(np.float32)
    img_gt_L = cv2.imread(images_gt_L, 1).astype(np.float32)
    img_gt_R = cv2.imread(images_gt_R, 1).astype(np.float32)

    return img_rgb_L, img_gt_L, img_gt_R

def fill_mask_holes(mask):
    '''Fill holes in the mask using morphological operations'''
    mask_int8 = (mask.astype(np.float) * 255).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask_filled = cv2.morphologyEx(mask_int8, cv2.MORPH_CLOSE, kernel)
    return mask_filled.astype(np.float) / 255

def save_images(blended_imgs, img_index, output_dir):
    sub_img_l, sub_img_r, sub_img_c, img_gt_l, img_gt_r, img_gt_c = blended_imgs
    img_name = f"{img_index}"

    if not os.path.exists(f"{output_dir}/dp_rainy"):
        os.makedirs(f"{output_dir}/dp_rainy")
    if not os.path.exists(f"{output_dir}/dp_gt"):
        os.makedirs(f"{output_dir}/dp_gt")

    cv2.imwrite(f"{output_dir}/dp_rainy/{img_name}_L.png", sub_img_l)
    cv2.imwrite(f"{output_dir}/dp_rainy/{img_name}_R.png", sub_img_r)
    cv2.imwrite(f"{output_dir}/dp_rainy/{img_name}_C.png", sub_img_c)
    cv2.imwrite(f"{output_dir}/dp_gt/{img_name}_GT_L.png", img_gt_l)
    cv2.imwrite(f"{output_dir}/dp_gt/{img_name}_GT_R.png", img_gt_r)
    cv2.imwrite(f"{output_dir}/dp_gt/{img_name}_GT_C.png", img_gt_c)