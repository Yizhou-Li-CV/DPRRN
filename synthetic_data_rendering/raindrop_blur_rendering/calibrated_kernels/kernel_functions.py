from pylab import plt
import numpy as np
import cv2

from scipy.signal import convolve2d


def padding_img(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def extract_patches(images, patch_size, num_rows, num_cols, padding=0):
    """ Divide images into image patches according to patch parameters
    Args:
    images: [..., #rows * P, #cols * P, C] height, width, #channels, P: patch size
    Returns:
    image_patches: [#rows * #cols, ..., P, P, C] The resulting image patches.
    """

    xv, yv = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
    yv *= patch_size
    xv *= patch_size

    patch_size_padding = patch_size + 2 * padding
    xv_size, yv_size = np.meshgrid(np.arange(patch_size_padding), np.arange(patch_size_padding))

    yv_all = yv.reshape(-1)[..., None, None] + yv_size[None, ...]
    xv_all = xv.reshape(-1)[..., None, None] + xv_size[None, ...]
    # print(np.max(yv_all), np.max(xv_all), images.shape)
    patches = images[yv_all, xv_all, :]
    patches = np.moveaxis(patches, -4, 0)

    return patches


def stitch_patches(patches, patch_size, num_rows, num_cols, stitch_axis):
    """ Stitch patches according to the given dimension
    Args:
    patches: [#rows * #cols, ..., P, P, C] / [#rows * #cols, ..., F, F]
    stitch_axis: (-3, -2) / (-2, -1)
    Returns:
    [..., #rows * P, #cols * P, C]  stitched images / [..., #rows * F, #cols * F] stitched kernels
    """

    axis_row, axis_col = stitch_axis
    patches_reshape = np.reshape(patches, (num_rows, num_cols, *patches.shape[1:]))
    patches_reshape = np.moveaxis(patches_reshape, (0, 1), (axis_row - 2, axis_col - 1))
    new_shape = np.array(patches.shape[1:])
    new_shape[axis_row] *= num_rows
    new_shape[axis_col] *= num_cols
    images = np.reshape(patches_reshape, new_shape)

    return images


def patch_wise_kernel_filter(img, gt_img, kernel_size):
    row = 6
    col = 8
    h, w, c = img.shape

    target_shape = (720 + 80, 540 + 60)
    resolution_scale = target_shape[0] / w
    current_kernel_size = kernel_size * resolution_scale

    # kernel here needs to be flipped
    ori_kernel_left = np.load('blur_kernels_left.npy')
    ori_kernel_right = np.load('blur_kernels_right.npy')

    k_ls = []
    k_rs = []
    k_cs = []

    for i in range(6*8):
        k_l, k_r, k_c = scale_kernel(current_kernel_size, ori_kernel_left[i], ori_kernel_right[i])
        k_ls.append(k_l)
        k_rs.append(k_r)
        k_cs.append(k_c)

    padding = k_ls[0].shape[-1] // 2

    # w, h
    # target_shape = (720 + 60, 540 + 60)
    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_LINEAR)
    gt_img = cv2.resize(gt_img, target_shape, interpolation=cv2.INTER_LINEAR)
    gt_img = gt_img[padding:padding+540, padding:padding+720]

    # print(img.shape)

    patch_size = 720 // col
    print(patch_size, k_ls[0].shape)

    patches = extract_patches(img, patch_size, row, col, padding=padding)
    print(patches.shape)

    filtered_patches_left = []
    filtered_patches_right = []
    filtered_patches_center = []

    for patch, k_l, k_r, k_c in zip(patches, k_ls, k_rs, k_cs):
        img_l = convolve2d(patch[..., 1], k_l, mode='valid')
        img_r = convolve2d(patch[..., 1], k_r, mode='valid')
        img_c = convolve2d(patch[..., 1], k_c, mode='valid')
        filtered_patches_left.append(img_l)
        filtered_patches_right.append(img_r)
        filtered_patches_center.append(img_c)

    print(filtered_patches_left[0].shape)

    filtered_left = stitch_patches(np.array(filtered_patches_left), patch_size, row, col, [-2, -1])
    filtered_right = stitch_patches(np.array(filtered_patches_right), patch_size, row, col, [-2, -1])
    filtered_center = stitch_patches(np.array(filtered_patches_center), patch_size, row, col, [-2, -1])

    target_w = int(filtered_left.shape[1] // resolution_scale)
    target_h = int(filtered_left.shape[0] // resolution_scale)

    print(filtered_left.shape, target_w, target_h)

    filtered_left = cv2.resize(filtered_left, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    filtered_right = cv2.resize(filtered_right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    filtered_center = cv2.resize(filtered_center, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    gt_img = cv2.resize(gt_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)[..., 1]

    return filtered_left, filtered_right, filtered_center, gt_img


def center_crop_with_padding(img, target_w, target_h, padding):
    h, w = img.shape[:2]
    start_h = (h - target_h) // 2 - padding
    start_w = (w - target_w) // 2 - padding

    return img[start_h:start_h+target_h+padding*2, start_w:start_w+target_w+padding*2]


def patch_wise_kernel_filter_real_rendered_kernel_varied(img_L, gt_img_L, img_R, gt_img_R, kernel_sizes):
    row = 6
    col = 8
    h, w, c = img_L.shape

    # max_kernel_size = np.max(kernel_sizes)
    target_w = 528
    target_h = 396

    # kernel here needs to be flipped
    ori_kernel_left = np.load('calibrated_kernels/blur_kernels_left.npy')
    ori_kernel_right = np.load('calibrated_kernels/blur_kernels_right.npy')

    k_ls = []
    k_rs = []
    k_cs = []

    print('kernel sizes:', kernel_sizes)

    for i in range(6*8):
        k_l, k_r, k_c = scale_kernel(kernel_sizes[i // 8], ori_kernel_left[i], ori_kernel_right[i])
        k_ls.append(k_l)
        k_rs.append(k_r)
        k_cs.append(k_c)

    vis_size = 35
    combined_kernels = np.zeros((vis_size*6, vis_size*8))
    # print('k_shape', ori_kernel_right.shape)
    for i in range(6):
        for j in range(8):
            k_r = k_rs[i*8 + j]
            h, w = k_r.shape
            if h > vis_size:
                k_r = center_crop_with_padding(k_r, vis_size, vis_size, padding=0)
            else:
                k_r = padding_img(k_r, vis_size, vis_size)
            combined_kernels[i*vis_size:(i+1)*vis_size, j*vis_size:(j+1)*vis_size] = k_r[::-1, ::-1]

    plt.imsave('scaled_kernels_left.png', combined_kernels)

    # use max kernel shape
    padding = k_ls[0].shape[-1] // 2
    
    # make sure patches are from center fov of the real dp image
    img_L = center_crop_with_padding(img_L, target_w=target_w, target_h=target_h, padding=padding)
    img_R = center_crop_with_padding(img_R, target_w=target_w, target_h=target_h, padding=padding)
    gt_img_L = center_crop_with_padding(gt_img_L, target_w=target_w, target_h=target_h, padding=padding)
    gt_img_R = center_crop_with_padding(gt_img_R, target_w=target_w, target_h=target_h, padding=padding)


    # w, h
    # target_shape = (720 + 60, 540 + 60)
    gt_img_L = gt_img_L[padding:padding + target_h, padding:padding + target_w]
    gt_img_R = gt_img_R[padding:padding + target_h, padding:padding + target_w]

    patch_size = 528 // col
    patches_L = extract_patches(img_L, patch_size, row, col, padding=padding)
    patches_R = extract_patches(img_R, patch_size, row, col, padding=padding)

    filtered_patches_left = []
    filtered_patches_right = []
    filtered_patches_center = []

    for patch_L, patch_R, k_l, k_r, k_c in zip(patches_L, patches_R, k_ls, k_rs, k_cs):

        # kernel should be exchanged or flipped as it is foreground be blurred, not the background be blurred
        k_l_front_defocus = k_r
        k_r_front_defocus = k_l
        # print(patch_L.shape)
        img_l = convolve2d(patch_L[..., 1], k_l_front_defocus, mode='valid')

        # during convolve2d, the kernel is horizontally/vertically flipped!!
        # during convolve2d, the kernel is horizontally/vertically flipped!!
        # during convolve2d, the kernel is horizontally/vertically flipped!!

        img_l = center_crop_with_padding(img_l, patch_size, patch_size, padding=0)
        img_r = convolve2d(patch_R[..., 1], k_r_front_defocus, mode='valid')
        img_r = center_crop_with_padding(img_r, patch_size, patch_size, padding=0)
        img_c = (img_l + img_r) / 2
        # print('L patch shape', img_l.shape)
        filtered_patches_left.append(img_l)
        filtered_patches_right.append(img_r)
        filtered_patches_center.append(img_c)

    # shape should be 396 * 528
    filtered_left = stitch_patches(np.array(filtered_patches_left), patch_size, row, col, [-2, -1])
    filtered_right = stitch_patches(np.array(filtered_patches_right), patch_size, row, col, [-2, -1])
    filtered_center = stitch_patches(np.array(filtered_patches_center), patch_size, row, col, [-2, -1])


    return filtered_left, filtered_right, filtered_center, gt_img_L, gt_img_R


def scale_kernel(tgt_kernel_size, ori_kernel_left, ori_kernel_right):

    default_size = 28
    scale = tgt_kernel_size / default_size
    tgt_kernel_shape = int(ori_kernel_left.shape[1] * scale)
    if tgt_kernel_shape % 2 == 0:
        tgt_kernel_shape += 1
    tgt_kernel_shape = [tgt_kernel_shape, tgt_kernel_shape]
    tgt_kernel_left = cv2.resize(ori_kernel_left, tgt_kernel_shape, interpolation=cv2.INTER_LINEAR)
    tgt_kernel_right = cv2.resize(ori_kernel_right, tgt_kernel_shape, interpolation=cv2.INTER_LINEAR)

    tgt_kernel_left /= np.sum(tgt_kernel_left)
    tgt_kernel_right /= np.sum(tgt_kernel_right)

    tgt_kernel_center = (tgt_kernel_left + tgt_kernel_right) / 2

    return tgt_kernel_left, tgt_kernel_right, tgt_kernel_center
