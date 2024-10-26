import numpy as np
from calibrated_kernels.kernel_functions import patch_wise_kernel_filter_real_rendered_kernel_varied
from utils import alpha_blending, gray_to_rgb, split_scenes, load_aif_file_info, load_images, fill_mask_holes, save_images, check_dir, get_dist_with_angle
import argparse

def get_params():
    '''Get camera parameters of Google Pixel 4'''
    return {
        'focus_dis': 10000,
        'focal_len': 27 / (43.2666 / 7.06),
        'f_stop': 1.7,
        'lens_width': 5.64,
        'lens_x_res': 800,
        'coc_max': 100,
        'fov': 30
    }

def raindrop_blur_rendering(cam_params, aif_dir_name, save_root_dir, scene_num, train_ratio):
    '''Raindrop blur rendering'''
    train_indexes, test_indexes = split_scenes(scene_num, train_ratio)
    file_info = load_aif_file_info(aif_dir_name)
    train_dir = f'{save_root_dir}/train'
    val_dir = f'{save_root_dir}/val'
    check_dir(train_dir)
    check_dir(val_dir)

    for i in train_indexes:
        process_image(i, file_info, aif_dir_name, cam_params, train_dir)

    for i in test_indexes:
        process_image(i, file_info, aif_dir_name, cam_params, val_dir)


def process_image(i, file_info, aif_dir_name, params, output_dir):
    '''Read AiF images and rendering DP blur'''
    if i not in file_info:
        print(f"No file info for index {i}")
        return

    raindrop_depth = file_info[i]['depth']
    theta_angle = file_info[i]['angle']
    file_name_L = file_info[i]['file_name_L']
    fov = params['fov']
    file_name_R = file_name_L.replace('L', 'R')

    print(f"Processing: {file_name_L}, Angle: {theta_angle}, Depth: {raindrop_depth}")

    images = load_images(aif_dir_name, file_name_L, file_name_R)
    if images is None:
        raise ValueError("Failed to load images")

    img_rgb_L, img_gt_L, img_gt_R = images
    img_rgb_R, raindrop_mask, alpha_mask = preprocess_images(img_rgb_L, img_gt_L, img_gt_R)

    depth_arr = calculate_depth_array(raindrop_depth, theta_angle, fov)
    kernel_sizes = calculate_kernel_sizes(depth_arr, params)

    sub_imgs, alpha_masks = apply_blur_filter(img_rgb_L, img_gt_L, img_rgb_R, img_gt_R, raindrop_mask, alpha_mask, kernel_sizes)
    
    blended_imgs = blend_images(sub_imgs, alpha_masks)

    save_images(blended_imgs, i, output_dir)


def preprocess_images(img_rgb_L, img_gt_L, img_gt_R):
    '''Create right image using the vignetting scale in non-rain GT images, 
    and get raindrop mask, and alpha mask for blending'''
    vignetting_scale = img_gt_R / img_gt_L
    img_rgb_R = np.clip(img_rgb_L * vignetting_scale, 0, 255)

    raindrop_mask = np.abs(img_rgb_R - img_gt_R) >= 1
    raindrop_mask = fill_mask_holes(raindrop_mask)

    alpha_mask = raindrop_mask.astype(np.float32)

    return img_rgb_R, raindrop_mask, alpha_mask


def calculate_depth_array(raindrop_depth, theta_angle, fov_angle=30):
    '''Calculate depth array based on the given raindrop depth and tilted angle'''
    top_depth, bottom_depth = get_dist_with_angle(raindrop_depth, fov_angle=fov_angle, theta_angle=theta_angle)
    # 6 rows of kernels are calibrated and used to blur the raindrops
    return np.linspace(top_depth, bottom_depth, num=6)

def calculate_kernel_sizes(depth_arr, params):
    '''Calculate kernel sizes based on the given depth array and camera parameters'''
    cocs = calculate_cocs(depth_arr, params)
    return [abs(round(coc / params['lens_width'] * params['lens_x_res'] * -2)) for coc in cocs]

def calculate_cocs(depth_arr, params):
    '''Calculate CoC size based on the given depth array and camera parameters'''
    cocs = []
    for depth in depth_arr:
        q = params['focal_len'] / params['f_stop']
        s_dash = params['focal_len'] * params['focus_dis'] / (params['focus_dis'] - params['focal_len'])
        r = q / 2 * s_dash / params['focus_dis'] * abs(depth - params['focus_dis']) / depth

        coc_size = min(abs(r), params['coc_max']) * np.sign(r)
        cocs.append(coc_size)
    return cocs

def apply_blur_filter(img_rgb_L, img_gt_L, img_rgb_R, img_gt_R, raindrop_mask, alpha_mask, kernel_sizes):
    '''Apply blur filter to the input images using the given kernel sizes'''
    sub_img_l, sub_img_r, sub_img_c, img_gt_l, img_gt_r = patch_wise_kernel_filter_real_rendered_kernel_varied(
        img_rgb_L * raindrop_mask, img_gt_L, img_rgb_R * raindrop_mask, img_gt_R, kernel_sizes)
    # apply blur to raindrop mask as well
    alpha_mask_l, alpha_mask_r, alpha_mask_c, _, _ = patch_wise_kernel_filter_real_rendered_kernel_varied(
        alpha_mask, alpha_mask, alpha_mask, alpha_mask, kernel_sizes)

    img_gt_c = (img_gt_l + img_gt_r) / 2

    return (sub_img_l, sub_img_r, sub_img_c, img_gt_l, img_gt_r, img_gt_c), (alpha_mask_l, alpha_mask_r, alpha_mask_c)

def blend_images(sub_imgs, alpha_masks):
    '''Blend the rain-free image and raindrop using the given alpha masks'''
    sub_img_l, sub_img_r, sub_img_c, img_gt_l, img_gt_r, img_gt_c = sub_imgs
    alpha_mask_l, alpha_mask_r, alpha_mask_c = alpha_masks

    sub_img_l = alpha_blending(gray_to_rgb(sub_img_l), img_gt_l, gray_to_rgb(alpha_mask_l))
    sub_img_r = alpha_blending(gray_to_rgb(sub_img_r), img_gt_r, gray_to_rgb(alpha_mask_r))
    sub_img_c = alpha_blending(gray_to_rgb(sub_img_c), img_gt_c, gray_to_rgb(alpha_mask_c))

    return sub_img_l, sub_img_r, sub_img_c, img_gt_l, img_gt_r, img_gt_c


def main(aif_dir_name, save_root_dir, scene_num, train_ratio):
    cam_params = get_params()
    raindrop_blur_rendering(cam_params, aif_dir_name, save_root_dir, scene_num=1, train_ratio=0.8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Raindrop blur rendering')
    parser.add_argument('--aif_dir_name', type=str, help='Path to AiF root path (with rainy and clean directories)', required=True)
    parser.add_argument('--save_root_dir', type=str, help='Path to save the output directory', required=True)
    parser.add_argument('--scene_num', type=int, help='Total number of rendered scenes (ignore different rain patterns, e.g. 2400/4=600 scenes if 2400 pairs rendered, with 4 rain patterns each scene)', required=True)
    parser.add_argument('--train_ratio', type=float, help='Train/val split ratio (0.8 for 80/20 split)', default=0.8)

    args = parser.parse_args()
    main(args.aif_dir_name, args.save_root_dir, args.scene_num, args.train_ratio)