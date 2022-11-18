import torch
import torch.nn.functional as F
from models.base_model import BaseModel
from network import build_networks
from util.standard_derain_metrics import SSIM_Derain_GPU, PSNR_Derain_GPU, BinaryMetrics
from util.loss_func import SSIM_Loss
from util.radam import RAdam


class SyntheticTrainTestModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch')

        if is_train:
            parser.add_argument('--lambda_ssim_image', type=float, default=1,
                                help='weight for negative ssim loss')
            parser.add_argument('--warmup_epochs', type=int, default=100,
                                help='the epoch number for DP raindrop detection warmup')
            parser.add_argument('--gradient_clipping', type=float, default=-1,
                                help='gradient clipping for lstm network')

        parser.add_argument('--mask_threshold', type=float, default=1e-3,
                            help='the threshold to select rain pixels')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses/metrics you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_left_mask']
        self.test_metric_names = ['ssim', 'psnr', 'dice', 'iou']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_L', 'real_R', 'fake_LB', 'fake_RB', 'fake_B', 'fake_mask_C_vis', 'gt_mask']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.SSIM_metric = SSIM_Derain_GPU().cuda()
        self.PSNR_metric = PSNR_Derain_GPU().cuda()

        if self.isTrain:
            self.warm_up_epoch = self.opt.warmup_epochs
        else:
            self.warm_up_epoch = 0

        self.netG = build_networks.define_G(opt.netG,
                                            opt.init_type, opt.init_gain,
                                            self.gpu_ids)

        self.criterionPixelAcc = BinaryMetrics()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = SSIM_Loss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.optimizer_G = RAdam(self.netG.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)

    def get_mask(self, rainy, clean):
        rain_map = torch.abs(rainy - clean)
        gt_mask = rain_map > 0
        gt_mask = gt_mask.float()
        gt_mask = torch.max(gt_mask, dim=1, keepdim=True)[0]

        return gt_mask

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_L = input['B_L'].to(self.device)
        self.real_B_R = input['B_R'].to(self.device)
        self.real_L = input['L'].to(self.device)
        self.real_R = input['R'].to(self.device)

        self.gt_mask_L = self.get_mask(self.real_L, self.real_B_L)
        self.gt_mask_R = self.get_mask(self.real_R, self.real_B_R)
        self.gt_mask = torch.maximum(self.gt_mask_L, self.gt_mask_R)

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

        if self.epoch != 'latest' and self.epoch <= self.warm_up_epoch:
            # print('Epoch', self.epoch)
            self.fake_masks = self.netG(*inputs, warm_up=True)

            self.fake_mask_L, self.fake_mask_R, self.fake_mask_C = self.fake_masks
            self.fake_mask_L_vis = F.sigmoid(self.fake_mask_L)
            self.fake_mask_R_vis = F.sigmoid(self.fake_mask_R)
            self.fake_mask_C_vis = F.sigmoid(self.fake_mask_C)

            self.fake_B = torch.zeros_like(self.real_L)
            self.fake_LB = torch.zeros_like(self.real_L)
            self.fake_RB = torch.zeros_like(self.real_L)
        else:
            self.fake_B, self.fake_mask_L, self.fake_mask_R, self.fake_mask_C, self.fake_BS = self.netG(*inputs, warm_up=False)
            self.fake_CB, self.fake_LB, self.fake_RB = self.fake_BS[:3]

            self.fake_mask_L_vis = F.sigmoid(self.fake_mask_L)
            self.fake_mask_R_vis = F.sigmoid(self.fake_mask_R)
            self.fake_mask_C_vis = F.sigmoid(self.fake_mask_C)

    # calculate metrics only used in printing (no grad)
    def cal_metrics(self):
        self.loss_ssim = self.SSIM_metric(self.fake_B, self.real_B)
        self.loss_psnr = self.PSNR_metric(self.fake_B, self.real_B)
        self.mask_metrics = self.criterionPixelAcc(self.gt_mask, self.fake_mask_C)
        self.loss_dice, self.loss_iou = self.mask_metrics

    # calculate training loss
    def cal_G_loss(self):

        self.loss_G = 0.
        self.loss_G_left_mask = 0
        self.loss_G_right_mask = 0

        if self.epoch <= self.warm_up_epoch:
            # print('epoch < 10')
            self.loss_G_left_mask = self.criterionBCE(self.fake_mask_L, self.gt_mask_L)
            self.loss_G_right_mask = self.criterionBCE(self.fake_mask_R, self.gt_mask_R)
            self.loss_G += self.loss_G_left_mask + self.loss_G_right_mask
        else:
            self.loss_G_left = -self.criterionSSIM((self.fake_LB + 1) / 2.,
                                                   (self.real_B_L + 1) / 2.)
            self.loss_G_right = -self.criterionSSIM((self.fake_RB + 1) / 2.,
                                                    (self.real_B_R + 1) / 2.)
            self.loss_G_center = -self.criterionSSIM((self.fake_CB + 1) / 2.,
                                                    (self.real_B + 1) / 2.)
            self.loss_G_fusion = -self.criterionSSIM((self.fake_B + 1) / 2.,
                                                     (self.real_B + 1) / 2.)
            self.loss_G_left_mask = self.criterionBCE(self.fake_mask_L, self.gt_mask_L)
            self.loss_G_right_mask = self.criterionBCE(self.fake_mask_R, self.gt_mask_R)
            self.loss_G += self.loss_G_fusion + self.loss_G_left + self.loss_G_right + self.loss_G_center + \
                           self.loss_G_left_mask + self.loss_G_right_mask

    def backward_G(self):
        self.loss_G.backward()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.cal_G_loss()
        self.backward_G()

        if self.opt.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.gradient_clipping)
        self.optimizer_G.step()
