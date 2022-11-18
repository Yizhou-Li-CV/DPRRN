from numpy.lib.stride_tricks import as_strided as ast

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp


def RGBTensor_to_Y(tensor):
    n, c, h, w = tensor.size()
    if n > 1:
        assert 'Test batch size is larger than 1!'

    # convert the tensor to an uint8 image
    img = tensor.detach()[0]
    img = (img + 1.) / 2. * 255.
    img = torch.clamp(img, min=0., max=255.)
    img = torch.floor(img)

    # convert the RGB image into Y, by simulating the RGB->Y process in Matlab
    img_gray = img[[0]] * 0.299 + img[[1]] * 0.587 + img[[2]] * 0.114
    img_gray = img_gray / 255 * 219 + 16
    img_gray = torch.unsqueeze(img_gray, dim=0)

    return img_gray


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


class PSNR_Derain_GPU(torch.nn.Module):

    def __init__(self):
        super(PSNR_Derain_GPU, self).__init__()

    def forward(self, img1, img2):
        img1 = RGBTensor_to_Y(img1)
        img2 = RGBTensor_to_Y(img2)

        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100

        PIXEL_MAX = 255.0
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class SSIM_Derain_GPU(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_Derain_GPU, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, data_range=1.0):

        if data_range != 255:
            img1 = RGBTensor_to_Y(img1) / 255.
            img2 = RGBTensor_to_Y(img2) / 255.

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, data_range)


class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        iou = (tp + self.eps) / (fp + tp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        return pixel_acc, dice, iou, precision, specificity, recall
        # return dice, iou

    def __call__(self, y_true, y_pred):
        # y_true: (N, 1, H, W)
        # y_pred: (N, 1, H, W)
        y_true = y_true[:, 0]
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = torch.nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = torch.nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        pixel_acc, dice, iou, precision, specificity, recall = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        # return [pixel_acc, dice, precision, specificity, recall]
        return [dice, iou]


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, data_range=1.0):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True, data_range=1.0):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, data_range)