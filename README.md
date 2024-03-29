## [Dual-Pixel Raindrop Removal, BMVC2022 (Oral)](https://arxiv.org/abs/2210.13321)
### Introduction
Removing raindrops in images has been addressed as a significant task for various computer vision applications. In this paper, we propose the first method using a Dual-Pixel (DP) sensor to better address the raindrop removal. Our key observation is that raindrops attached to a glass window yield noticeable disparities in DP's left-half and right-half images, while almost no disparity exists for in-focus backgrounds. Therefore, DP disparities can be utilized for robust raindrop detection. The DP disparities also brings the advantage that the occluded background regions by raindrops are shifted between the left-half and the right-half images. Therefore, fusing the information from the left-half and the right-half images can lead to more accurate background texture recovery. Based on the above motivation, we propose a DP Raindrop Removal Network (DPRRN) consisting of DP raindrop detection and DP fused raindrop removal. To efficiently generate a large amount of training data, we also propose a novel pipeline to add synthetic raindrops to real-world background DP images. Experimental results on synthetic and real-world datasets demonstrate that our DPRRN outperforms existing state-of-the-art methods, especially showing better robustness to real-world situations. Check our project page at http://www.ok.sc.e.titech.ac.jp/res/SIR/dprrn/dprrn.html.

## Prerequisites
- Python 3.9, PyTorch >= 1.8.0
- Requirements: opencv-python, tensorflow 1.x/2.x (for use of tensorboard)

## Dataset Preparation
- Synthetic-Raindrop Datasets: 1960 training pairs and 492 test pairs

- Real-world Dataset: 82 test pairs

Both can be downloaded from [[GoogleDrive]](https://drive.google.com/drive/folders/1L3sXsthCAkBF_mI9K8VI8xwIDYWou7Wj?usp=sharing). Unzip both files to ./datasets for training and test.

The code of data rendering will be publicly available later.

## Training

Execute the command below to start training.
```
$ tensorboard --logdir logs/tensorboard/image_deraining
$ sh scripts/train_syn.sh
```
The tensorboard logs can be found at ./logs/, while trained models can be found at ./checkpoints/.

## Testing
Test with our pretrained models.
```
$ sh script/test_real_pretrained.sh
$ sh script/test_syn_pretrained.sh
```
Or test with your retrained models.
```
$ sh script/test_real_retrained.sh
$ sh script/test_syn_retrained.sh
```
The PSNR, SSIM and average inference time will be printed, and derained results are saved in the folder "./results/".

## Acknowledgement 
Code framework borrows from [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by [Jun-Yan Zhu](https://github.com/junyanz/). Thanks for sharing !


## Citation

```
@inproceedings{Li_2022_BMVC,
author    = {Yizhou Li and Yusuke Monno and Masatoshi Okutomi},
title     = {Dual-Pixel Raindrop Removal},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0439.pdf}
}
```
