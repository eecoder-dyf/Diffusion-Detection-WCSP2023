import torch
import sys 
from .wavelet import LiftingScheme2D
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from .swin import WaveletSwinTransformerBlock
import utils

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
        self.wavelet1 = LiftingScheme2D(in_planes=3, share_weights=True)
        self.wavelet2 = LiftingScheme2D(in_planes=backbone.conv1.out_channels, share_weights=True)
        self.wavelet3 = LiftingScheme2D(in_planes=backbone.layer2[0].conv1.in_channels, share_weights=True)
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
        LL1 = self.convll1(LL1)
        
        feat2 = self.backbone1(self.conv01(torch.cat([feat1, attn1], dim=1)))
        
        _, _, LL2, LH2, HL2, HH2 = self.wavelet2(LL1)
        attn2 = self.attn2(LH2, HL2, HH2, feat2)
        wavefeat2 = self.backbone23(attn2)
        LL2 = self.convll2(LL2)

        feat3 = self.backbone2(self.conv12(torch.cat([feat2, attn2], dim=1)))
        
        _, _, LL3, LH3, HL3, HH3 = self.wavelet3(LL2)
        attn3 = self.attn3(LH3, HL3, HH3, feat3)
        wavefeat3 = self.backbone33(attn3)

        feat4 = self.backbone3(self.conv23(torch.cat([feat3, attn3], dim=1)))
        
        feat5 = self.catlayer(torch.cat([wavefeat1, wavefeat2, wavefeat3, feat4], dim=1))
        feat5 = self.backbone4(feat5)
        fc_in = self.avg_pool(feat5)
        fc_in = torch.flatten(fc_in, 1)
        fc_out = self.fc(fc_in)

        if tsne:
            return fc_out, {"wavefeat1":wavefeat1, "wavefeat2":wavefeat2, "wavefeat3":wavefeat3, "feat4":feat4}
        else:
            return fc_out
