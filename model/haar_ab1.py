import torch
import sys 
from .wavelet import LiftingScheme2D
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from .swin import WaveletSwinTransformerBlock

class DWT2D(nn.Module):
    def __init__(self):
        super(DWT2D, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        c = x01 + x02
        d = x01 - x02
        LL = x1 + x2 + x3 + x4
        HL = -x1 - x2 + x3 + x4
        LH = -x1 + x2 - x3 + x4
        HH = x1 - x2 - x3 + x4
        return (c, d, LL, LH, HL, HH)

class Network(nn.Module):
    def __init__(self, backbone, mask=False):
        super(Network, self).__init__()
        self.backbone0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        if mask:
            self.backbone0[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone1 = nn.Sequential(backbone.maxpool,backbone.layer1)
        self.backbone2 = backbone.layer2
        self.backbone3 = backbone.layer3
        self.backbone4 = backbone.layer4
        self.wavelet1 = DWT2D()

        self.attn1 = WaveletSwinTransformerBlock(in_ch_wave=3, in_ch_spat=backbone.conv1.out_channels, dim=backbone.conv1.out_channels, num_heads=4)

        self.conv01 = nn.Conv2d(in_channels=2*backbone.layer1[0].conv1.in_channels,
                                out_channels=backbone.layer1[0].conv1.in_channels,
                                kernel_size=1, stride=1, padding=0)

        self.backbone13 = nn.Sequential(deepcopy(self.backbone1), deepcopy(self.backbone2), deepcopy(self.backbone3))

        self.catlayer = nn.Conv2d(in_channels=2*backbone.layer4[0].conv1.in_channels, out_channels=backbone.layer4[0].conv1.in_channels, kernel_size=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=2, bias=True)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x, tsne=False):
        image = x[:,0:3,:,:]
        _, _, LL1, LH1, HL1, HH1 = self.wavelet1(image)

        feat1 = self.backbone0(x)
        attn1 = self.attn1(LH1, HL1, HH1, feat1)
        wavefeat1 = self.backbone13(attn1)
        
        feat2 = self.backbone1(self.conv01(torch.cat([feat1, attn1], dim=1)))
        feat3 = self.backbone2(feat2)
        feat4 = self.backbone3(feat3)
        feat5 = self.catlayer(torch.cat([wavefeat1, feat4], dim=1))
        feat5 = self.backbone4(feat5)
        fc_in = self.avg_pool(feat5)
        fc_in = torch.flatten(fc_in, 1)
        fc_out = self.fc(fc_in)

        if tsne:
            return fc_out, {"wavefeat1":wavefeat1, "feat4":feat4}
        else:
            return fc_out
