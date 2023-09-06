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
        self.wavelet2 = DWT2D()
        self.wavelet3 = DWT2D()
        self.attn1 = WaveletSwinTransformerBlock(in_ch_wave=3, in_ch_spat=backbone.conv1.out_channels, dim=backbone.conv1.out_channels, num_heads=4)
        self.attn2 = WaveletSwinTransformerBlock(in_ch_wave=backbone.conv1.out_channels,
                                      in_ch_spat=backbone.layer2[0].conv1.in_channels,
                                      dim=backbone.layer2[0].conv1.in_channels, num_heads=8)
        self.attn3 = WaveletSwinTransformerBlock(in_ch_wave=backbone.layer2[0].conv1.in_channels,
                                      in_ch_spat=backbone.layer3[0].conv1.in_channels,
                                      dim=backbone.layer3[0].conv1.in_channels, num_heads=16)
        self.conv01 = nn.Conv2d(in_channels=2*backbone.layer1[0].conv1.in_channels,
                                out_channels=backbone.layer1[0].conv1.in_channels,
                                kernel_size=1, stride=1, padding=0)
        self.conv12 = nn.Conv2d(in_channels=2*backbone.layer2[0].conv1.in_channels,
                                out_channels=backbone.layer2[0].conv1.in_channels,
                                kernel_size=1, stride=1, padding=0)
        self.conv23 = nn.Conv2d(in_channels=2*backbone.layer3[0].conv1.in_channels,
                                out_channels=backbone.layer3[0].conv1.in_channels,
                                kernel_size=1, stride=1, padding=0)
        self.backbone13 = nn.Sequential(deepcopy(self.backbone1), deepcopy(self.backbone2), deepcopy(self.backbone3))
        self.backbone23 = nn.Sequential(deepcopy(self.backbone2), deepcopy(self.backbone3))
        self.backbone33 = deepcopy(self.backbone3)
        self.convll1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=backbone.conv1.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=backbone.conv1.out_channels),
            nn.ReLU(inplace=True)
        )
        self.convll2 = nn.Sequential(
            nn.Conv2d(in_channels=backbone.conv1.out_channels, out_channels=backbone.layer2[0].conv1.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=backbone.layer2[0].conv1.in_channels),
            nn.ReLU(inplace=True)
        )
        self.catlayer = nn.Conv2d(in_channels=4*backbone.layer4[0].conv1.in_channels, out_channels=backbone.layer4[0].conv1.in_channels, kernel_size=1)
        
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
        HH1 = self.convll1(HH1)
        
        feat2 = self.backbone1(self.conv01(torch.cat([feat1, attn1], dim=1)))
        # feat2 = self.backbone1(feat1 + attn1)
        _, _, LL2, LH2, HL2, HH2 = self.wavelet2(HH1)
        attn2 = self.attn2(LH2, HL2, HH2, feat2)
        wavefeat2 = self.backbone23(attn2)
        HH2 = self.convll2(HH2)

        feat3 = self.backbone2(self.conv12(torch.cat([feat2, attn2], dim=1)))
        # feat3 = self.backbone2(feat2 + attn2)
        _, _, LL3, LH3, HL3, HH3 = self.wavelet3(HH2)
        attn3 = self.attn3(LH3, HL3, HH3, feat3)
        wavefeat3 = self.backbone33(attn3)

        feat4 = self.backbone3(self.conv23(torch.cat([feat3, attn3], dim=1)))
        # feat4 = self.backbone3(feat3 + attn3)
        feat5 = self.catlayer(torch.cat([wavefeat1, wavefeat2, wavefeat3, feat4], dim=1))
        feat5 = self.backbone4(feat5)
        fc_in = self.avg_pool(feat5)
        fc_in = torch.flatten(fc_in, 1)
        fc_out = self.fc(fc_in)
        if tsne:
            return fc_out, {"wavefeat1":wavefeat1, "wavefeat2":wavefeat2, "wavefeat3":wavefeat3, "feat4":feat4}
        return fc_out
