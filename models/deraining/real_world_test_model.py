import torch
import torch.nn.functional as F
from models.base_model import BaseModel
from network import build_networks
from util.standard_derain_metrics import SSIM_Derain_GPU, PSNR_Derain_GPU, BinaryMetrics


class RealWorldTestModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        self.test_metric_names = ['ssim', 'psnr']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_L', 'real_R', 'fake_LB', 'fake_RB', 'fake_B', 'fake_mask_C_vis']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']

        self.SSIM_metric = SSIM_Derain_GPU().cuda()
        self.PSNR_metric = PSNR_Derain_GPU().cuda()

        self.netG = build_networks.define_G(opt.netG,
                                            opt.init_type, opt.init_gain,
                                            self.gpu_ids)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_L = input['L'].to(self.device)
        self.real_R = input['R'].to(self.device)
        self.image_paths = input['A_paths']

    def get_list_last_item(self, input):
        if isinstance(input, list):
            return input[-1]
        else:
            return input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # print(self.epoch)
        inputs = [self.real_L, self.real_R]

        self.fake_B, self.fake_mask_L, self.fake_mask_R, self.fake_mask_C, self.fake_BS = self.netG(*inputs, warm_up=False)
        self.fake_CB, self.fake_LB, self.fake_RB = self.fake_BS[:3]

        self.fake_mask_L_vis = F.sigmoid(self.fake_mask_L)
        self.fake_mask_R_vis = F.sigmoid(self.fake_mask_R)
        self.fake_mask_C_vis = F.sigmoid(self.fake_mask_C)

    # calculate loss only used in printing (no grad)
    def cal_metrics(self):
        self.loss_ssim = self.SSIM_metric(self.fake_B, self.real_B)
        self.loss_psnr = self.PSNR_metric(self.fake_B, self.real_B)

    def cal_G_loss(self):
        pass

    def backward_G(self):
        pass

    def optimize_parameters(self):
        pass
