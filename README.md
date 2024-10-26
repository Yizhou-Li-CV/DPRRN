## [Dual-Pixel Raindrop Removal, BMVC2022, TPAMI2024](https://ieeexplore.ieee.org/document/10636073/?denied=)
### Introduction
Removing raindrops in images has been addressed as a significant task for various computer vision applications. In this paper, we propose the first method using a Dual-Pixel (DP) sensor to better address the raindrop removal. Our key observation is that raindrops attached to a glass window yield noticeable disparities in DP's left-half and right-half images, while almost no disparity exists for in-focus backgrounds. Therefore, DP disparities can be utilized for robust raindrop detection. The DP disparities also brings the advantage that the occluded background regions by raindrops are shifted between the left-half and the right-half images. Therefore, fusing the information from the left-half and the right-half images can lead to more accurate background texture recovery. Based on the above motivation, we propose a DP Raindrop Removal Network (DPRRN) consisting of DP raindrop detection and DP fused raindrop removal. To efficiently generate a large amount of training data, we also propose a novel pipeline to add synthetic raindrops to real-world background DP images. Experimental results on synthetic and real-world datasets demonstrate that our DPRRN outperforms existing state-of-the-art methods, especially showing better robustness to real-world situations. Check our project page at http://www.ok.sc.e.titech.ac.jp/res/SIR/dprrn/dprrn.html.

## Prerequisites
- Python 3.9, PyTorch >= 1.8.0
- Requirements: opencv-python, tensorflow 1.x/2.x (for use of tensorboard)

## Dataset Preparation
Synthetic-Raindrop Datasets: 
- Tilted: 1960 training pairs and 492 test pairs
- Vertical: 1960 training pairs and 492 test pairs

Real-waterdrop dataset: 
- Tilted: 130 test pairs
- Vertical: 125 test pairs

Real-car
- Tilted: 94 pairs

Download:
- BMVC ver. dataset: [[GoogleDrive]](https://drive.google.com/drive/folders/1L3sXsthCAkBF_mI9K8VI8xwIDYWou7Wj?usp=sharing).
- TPAMI ver. dataset: [[GoogleDrive]](https://drive.google.com/drive/folders/1-1V4ll1x5mqViI1zPUfufZ0GXMgR_s-r?usp=sharing).

 Unzip both files to ./datasets for training and test.

## Synthetic Dataset Rendering
Codes are in folder of ./synthetic_data_rendering/.

Please check the corresponding .py files for parameter explanation.
### Process DP Raw Captured by Google Pixel 4
```
$ python process_dp_raw.py --raw_dir xx --transferred_dir xx
```

### Render All-in-Focus Raindrops on DP Images
```
$ cd aif_raindrop_rendering
$ python render.py --file_dir xx --nb_imgs xx --save_dir xx
```

### Rendering DP Blur in All-in-Focus Raindrop Images
```
$ cd ../raindrop_blur_rendering
$ python render.py --aif_dir_name xx --save_root_dir xx --scene_num xx
```

## Training

Execute the command below to start training.
```
$ tensorboard --logdir logs/tensorboard/image_deraining
$ sh scripts/train_syn.sh
```
The tensorboard logs can be found at ./logs/, while trained models can be found at ./checkpoints/.

## Testing
Test with our pretrained models (TPAMI Verision is updated in ./checkpoints/).
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
