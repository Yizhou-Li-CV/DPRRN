import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class DPLRSyntheticDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'dp_rainy') # get the rainy image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'dp_gt')  # get the rain-free image directory
        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)  # get image paths
        self.B_paths = [path for path in self.B_paths if "GT_C" in path]
        self.B_paths = sorted(self.B_paths)

    def __getitem__(self, index):

        B_path = self.B_paths[index]
        # B_path = self.B_paths[index]
        # B_img_name = B_path.split('/')[-1]
        B_img_name = B_path.split('\\')[-1]
        B_img_L_name = B_img_name.replace('C', 'L')
        B_img_L_path = os.path.join(self.dir_B, B_img_L_name)
        B_img_R_name = B_img_name.replace('C', 'R')
        B_img_R_path = os.path.join(self.dir_B, B_img_R_name)
        A_img_name = B_img_name.replace('_GT', '')
        A_path = os.path.join(self.dir_A, A_img_name)
        L_img_name = B_img_name.replace('GT_C', 'L')
        L_path = os.path.join(self.dir_A, L_img_name)
        R_img_name = B_img_name.replace('GT_C', 'R')
        R_path = os.path.join(self.dir_A, R_img_name)
        if not os.path.exists(A_path):
            A_path = os.path.join(self.dir_A, B_img_name)
        A, B = Image.open(A_path).convert('RGB'), Image.open(B_path).convert('RGB')
        B_L, B_R = Image.open(B_img_L_path).convert('RGB'), Image.open(B_img_R_path).convert('RGB')
        # print('join:', os.path.join(self.dir_A, L_img_name), self.dir_A, L_img_name)
        L, R = Image.open(L_path).convert('RGB'), Image.open(R_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)
        L_transform = get_transform(self.opt, transform_params)
        R_transform = get_transform(self.opt, transform_params)

        A = A_transform(A)
        B = B_transform(B)
        B_L = B_transform(B_L)
        B_R = B_transform(B_R)
        L = L_transform(L)
        R = R_transform(R)

        return {'A': A, 'B': B, 'B_L': B_L, 'B_R': B_R, 'L': L, 'R': R, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)
