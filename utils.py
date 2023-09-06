import os
import sys
import torch
import argparse
import shutil
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    return v.lower() in ('true')

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Example training script')
    parser.add_argument(
        '-f',
        '--folder',
        type=str,
        default="/mnt/d/diffusion_faces/dataset",
        help='Dataset folder'
    )
    parser.add_argument(
        '--real',
        type=str,
        default='celeba_hq_256',
        help="Real dataset name"
    )
    parser.add_argument(
        '--fake',
        type=str,
        default='ldm_celeba256_200',
        help="Real dataset name"
    )
    parser.add_argument(
        '--cuda',
        type=int,
        nargs='+',
        default=[0],
        help='cuda device number'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='batch size'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=1000,
        help='epoch nums'
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        default=1e-4,
        type=float,
        help='Learning rate (default: %(default)s)'
    )
    parser.add_argument(
        '--mask',
        type=str2bool,
        default=False,
        help='use face mask or not'
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config/test.yaml",
        help='use face mask or not'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save model to disk'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Only test'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='checkpoint_best_loss.pth.tar',
        help='load weights'
    )
    parser.add_argument(
        '--tsne',
        action='store_true',
        help='calculate tsne map'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        required=False,
        help='pretrained weights'
    )

    args = parser.parse_args(argv)
    return args

def save_checkpoint(state, is_best, filename='checkpoint_best_loss.pth.tar'):
    if is_best:
        if os.path.exists(filename):
            shutil.copy(filename, filename+'.copy') #做好备份，防止写盘时突然断开导致模型文件损坏
        torch.save(state, filename)
        if os.path.exists(filename +".copy"):
            os.remove(filename+".copy")

def get_params(model, name='net'):
    print('The parameters of model ' + name)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def get_params_flops(tensor, network):
    from thop import profile, clever_format
    flops, params = profile(network, inputs=(tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print("Total parameters is {}, total flops is {}".format(params, flops))

def save_tensor_image(image, name):
    image_save = image[0].clone().cpu()
    unnorm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[2., 2., 2.]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   ])
    image_save = unnorm(image_save)
    image_save = torchvision.transforms.ToPILImage()(image_save)
    image_save = torchvision.transforms.Grayscale(1)(image_save)
    image_save.save(name)

def freeze_model(model, to_freeze_dict, keep_step=None):

    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass

    return model