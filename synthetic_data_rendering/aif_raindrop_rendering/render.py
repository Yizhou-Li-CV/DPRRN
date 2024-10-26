from raindrop_renderer import RaindropRenderer
import numpy as np
import ctypes
from multiprocessing import Process, Array, Value
import copy
import time
from PIL import Image
import os
import argparse


def get_image_for_mp(img):
    h, w = img.shape
    img_copy = copy.deepcopy(img)

    img_copy_mp = img_copy.reshape(h * w)
    img_mp = img.reshape(h * w)
    img_copy_mp = Array(ctypes.c_double, img_copy_mp)
    img_mp = Array(ctypes.c_double, img_mp)

    return img_mp, img_copy_mp


def render_raindrop_multiprocess_LR(L_image, R_image, params, renderer=None, nb_threads=6):
    h, w = L_image.shape
    if renderer is None:
        renderer = RaindropRenderer(params)
        renderer.get_sphere_raindrop_physics(w, h)

    x_range_total = np.array(range(0, w))
    xs = np.array_split(x_range_total, nb_threads)

    L_image_mp, L_rain_image_rendered_mp = get_image_for_mp(L_image)
    R_image_mp, R_rain_image_rendered_mp = get_image_for_mp(R_image)

    mask = np.zeros((h, w))
    mask_mp = mask.reshape(h * w)
    mask_mp = Array(ctypes.c_double, mask_mp)

    counter_mp = Value('i', 0)

    threads = []

    t1 = time.time()

    for x_range in xs:
        t = Process(target=renderer.render_multiprocess_LR,
                    args=(L_image_mp, L_rain_image_rendered_mp, R_image_mp, R_rain_image_rendered_mp, mask_mp, x_range, h, w, 'sphere', counter_mp))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    t2 = time.time()
    print('render time:', t2 - t1)

    L_rain_image_rendered = np.array(L_rain_image_rendered_mp).reshape((h, w))
    R_rain_image_rendered = np.array(R_rain_image_rendered_mp).reshape((h, w))
    mask = np.array(mask_mp).reshape((h, w))

    return L_rain_image_rendered, R_rain_image_rendered, renderer.psi / np.pi * 180, \
           renderer.density, renderer.glass_r, renderer.M, mask


def read_image_PIL(file_path, resize_shape):
    bg_image = Image.open(file_path)
    bg_image = bg_image.resize(resize_shape, Image.BILINEAR)
    bg_image = np.array(bg_image)[..., 0] / 255.0

    return bg_image


def gray_to_rgb(img):
    img = np.stack([img, img, img], axis=-1)
    return img


def save_numpy_image(img, path):
    img = gray_to_rgb(img)
    print(img.shape)
    img = np.round(img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

def main(file_dir, idx_range, rainy_dir, clean_dir, sample_for_each_img=4, nb_threads=32):
    if not os.path.exists(rainy_dir):
        os.makedirs(rainy_dir)
        os.makedirs(clean_dir)
    params = {}
    params['M_range'] = (50, 250)
    params['B_range'] = (4000, 8000)
    params['psi_range'] = (30, 45)
    params['tau_range'] = (30, 45)
    params['density_range'] = (2, 5)

    resize_shape = (800, 600)
    scale_factor = 2016 / resize_shape[0]

    params['fx'] = 1.621592900425656e+03 / scale_factor
    params['fy'] = 1.621023833056072e+03 / scale_factor
    params['cx'] = 1.020306839321437e+03 / scale_factor
    params['cy'] = 7.452945331082154e+02 / scale_factor

    idx_cnt = 0

    for idx in idx_range:
        L_img_path = os.path.join(file_dir, f'{idx}_L.png')
        R_img_path = os.path.join(file_dir, f'{idx}_R.png')

        L_img = read_image_PIL(L_img_path, resize_shape)
        R_img = read_image_PIL(R_img_path, resize_shape)

        for _ in range(sample_for_each_img):
            renderer = RaindropRenderer(params)
            renderer.get_sphere_raindrop_physics(*resize_shape)
            L_rain_image_rendered, R_rain_image_rendered, psi_angle, density, glass_r, depth, _ = render_raindrop_multiprocess_LR(L_img, R_img, params, renderer, nb_threads=nb_threads)

            density_str = "%.2f" % density
            radius_str = "%.2f" % glass_r
            save_numpy_image(L_rain_image_rendered, os.path.join(rainy_dir, f'{idx_cnt}_{int(psi_angle)}_{density_str}_{radius_str}_{depth}_L.png'))
            save_numpy_image(R_rain_image_rendered, os.path.join(rainy_dir, f'{idx_cnt}_{int(psi_angle)}_{density_str}_{radius_str}_{depth}_R.png'))
            save_numpy_image(L_img, os.path.join(clean_dir, f'{idx_cnt}_{int(psi_angle)}_{density_str}_{radius_str}_{depth}_L.png'))
            save_numpy_image(R_img, os.path.join(clean_dir, f'{idx_cnt}_{int(psi_angle)}_{density_str}_{radius_str}_{depth}_R.png'))

            idx_cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate raindrop images with physics simulation')
    parser.add_argument('--file_dir', type=str, required=True, help='Path to the directory containing left and right images')
    parser.add_argument('--nb_imgs', type=int, required=True, help='number of DP images to be processed')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the generated images')
    parser.add_argument('--sample_for_each_img', type=int, default=4, help='Number of raindrop patterns for each image')
    parser.add_argument('--nb_threads', type=int, default=16, help='Number of threads for parallel processing (based on CPU cores number)')
    args = parser.parse_args()
        
    idx_range = range(0, args.nb_imgs)
    rainy_dir = os.path.join(args.save_dir, 'rainy')
    clean_dir = os.path.join(args.save_dir, 'clean')

    main(args.file_dir, idx_range, rainy_dir, clean_dir, args.sample_for_each_img, args.nb_threads)
