from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class RB(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, down_ch=False):
        super(RB, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))

        if down_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu_out = relu(*param)

    def forward(self, x):

        identity = x

        x = self.conv(x)
        x = x + self.shortcut(identity)
        x = self.relu_out(x)

        return x


class Up(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, bilinear=True):
        super(Up, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_ch),
                relu(*param)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                norm_layer(out_ch),
                relu(*param)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=False,
                 upsampling='bilinear', n1=32, input_nc=9, output_nc=3, start7kernel=False):
        super(UNet, self).__init__()

        if input_nc is None:
            input_nc = 3

        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool1 = pool_layer(kernel_size=2, stride=2)
        self.Maxpool2 = pool_layer(kernel_size=2, stride=2)
        self.Maxpool3 = pool_layer(kernel_size=2, stride=2)

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        if not start7kernel:
            self.Conv_input = nn.Sequential(
                nn.Conv2d(input_nc, self.filters[0], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(self.filters[0]),
                relu(*param))
        else:
            self.Conv_input = nn.Sequential(nn.Conv2d(input_nc, self.filters[0], kernel_size=7, stride=1, padding=3),
                                            nn.InstanceNorm2d(self.filters[0]),
                                            relu(*param))

        self.Conv1 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=False)
        self.Conv2 = RB(self.filters[0], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Conv3 = RB(self.filters[1], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Conv4 = RB(self.filters[2], self.filters[3], norm_layer, leaky=leaky, down_ch=True)

        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')

        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv4_2 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')

        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv3_2 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')

        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv2_2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.filters[0], output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        x = input

        x_in = x

        x_in = self.Conv_input(x_in)

        e1 = self.Conv1(x_in)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4) + self.Up_conv4_2(e3)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3) + self.Up_conv3_2(e2)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2) + self.Up_conv2_2(e1)

        out_res = self.Conv(d2)

        x = out_res

        return x


class DPRRN(nn.Module):
    def __init__(self):
        super(DPRRN, self).__init__()

        self.UNet_LR_Loc = UNet(norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=False,
                                upsampling='bilinear', n1=32, input_nc=6, output_nc=1)
        self.UNet_RL_Loc = UNet(norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=False,
                                upsampling='bilinear', n1=32, input_nc=6, output_nc=1)

        self.UNet_L = UNet(norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=False,
                                  upsampling='bilinear', n1=32, input_nc=4)
        self.UNet_R = UNet(norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=False,
                                  upsampling='bilinear', n1=32, input_nc=4)

        self.UNet_fusion = UNet(norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=False,
                                upsampling='bilinear', n1=64, input_nc=10)

    def forward(self, in_left, in_right, warm_up=False):

        input_LR_Mask = torch.cat([in_left, in_right], dim=1)
        L_rain_mask = self.UNet_LR_Loc(input_LR_Mask)
        input_RL_Mask = torch.cat([in_right, in_left], dim=1)
        R_rain_mask = self.UNet_RL_Loc(input_RL_Mask)
        C_rain_mask = torch.maximum(L_rain_mask, R_rain_mask)

        if warm_up:
            # without sigmoid for BCE loss.
            return L_rain_mask, R_rain_mask, C_rain_mask

        L_rain_mask_sig = torch.sigmoid(L_rain_mask)
        R_rain_mask_sig = torch.sigmoid(R_rain_mask)
        C_rain_mask_sig = torch.sigmoid(C_rain_mask)

        input_L_derain = torch.cat([in_left, L_rain_mask_sig], dim=1)
        out_L = in_left - self.UNet_L(input_L_derain)
        input_R_derain = torch.cat([in_right, R_rain_mask_sig], dim=1)
        out_R = in_right - self.UNet_R(input_R_derain)
        out_C = (out_L + out_R) / 2

        input_fusion = torch.cat([out_C, out_L, out_R, C_rain_mask_sig], dim=1)
        out_final = out_C - self.UNet_fusion(input_fusion)

        return out_final, L_rain_mask, R_rain_mask, C_rain_mask, [out_C, out_L, out_R]
