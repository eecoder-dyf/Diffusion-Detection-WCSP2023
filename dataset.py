import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join as pj
import glob
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import yaml

class FakeDataset(Dataset):
    def __init__(self, folder, real_data, fake_data, resoultion=(256,256), normalize=False, phase='train', 
                 mask=False, test_config="None"):
        real_files = glob.glob(pj(folder, real_data, phase, "*"))
        fake_files = glob.glob(pj(folder, fake_data, phase, "*"))
        real_labels = [1] * len(real_files)
        fake_labels = [0] * len(fake_files)
        real_files = list(zip(real_files, real_labels))
        fake_files = list(zip(fake_files, fake_labels))
        self.files = real_files + fake_files
        random.shuffle(self.files)
        self.H = resoultion[0]
        self.W = resoultion[1]
        self.normalize = normalize
        self.phase = phase
        self.mask = mask
        self.folder = folder
        if os.path.exists(test_config):
            configfile = open(test_config, encoding='utf-8')
            config = yaml.load(configfile, Loader=yaml.FullLoader)
            print(config)
            self.jpeg = config['jpeg']
            self.blur = config['blur']
            self.g_noise = config["g-noise"]
            self.sp_noise = config["sp-noise"]

    def __len__(self):
        return len(self.files)
    
    def jpeg_compress(self, image, quality):
        if quality > 0:
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode(".jpg", image, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            decimg = Image.fromarray(cv2.cvtColor(decimg,cv2.COLOR_BGR2RGB))
            return decimg
        else:
            return image

    
    def transforms(self, image):
        if self.normalize:
            if self.phase == "train":
                transform = transforms.Compose([
                    transforms.Resize((self.H, self.W)),
                    transforms.ColorJitter(brightness=0.5),
                    transforms.ColorJitter(contrast=0.5),
                    transforms.ColorJitter(saturation=0.5),
                    transforms.ColorJitter(hue=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((self.H, self.W)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.H, self.W)),
                transforms.ToTensor(),
            ])
        return transform(image)
    
    def Blur(self, image):
        if random.uniform(0, 1) < self.blur["p"]:
            kernel_size = self.blur["kernel-size"]
            sigma = self.blur["sigma"]
            add_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            return add_blur(image)
        else:
            return image

    def gaussion_noise(self, image):
        add_gau_noise = AddGaussianNoise(p=self.g_noise["p"], variance=self.g_noise["variance"])
        return add_gau_noise(image)

    def saltpepper_noise(self, image):
        add_sp_noise = AddSaltPepperNoise(p=self.sp_noise["p"], density=self.sp_noise["density"])
        return add_sp_noise(image)    
    
    def __getitem__(self, index):
        image = Image.open(self.files[index][0])
        # if self.phase == "test":
        image = self.Blur(image)
        image = self.gaussion_noise(image)
        image = self.saltpepper_noise(image)
        image = self.jpeg_compress(image, self.jpeg)
        
        image = self.transforms(image)

        label = self.files[index][1]
        if label == 0:
            label = torch.tensor([0,1], dtype=torch.float32)
        else:
            label = torch.tensor([1,0], dtype=torch.float32)
        if self.mask:
            name = self.files[index][0].split("/")[-1]
            dataset = self.files[index][0].split("/")[-3]
            # mask_root = os.path.join(self.folder, dataset, "mask", self.phase, name)
            mask_root = os.path.join(self.folder, dataset, self.phase, name)
            mask_image = Image.open(mask_root)

            mask_image = self.Blur(mask_image)
            mask_image = self.gaussion_noise(mask_image)
            mask_image = self.saltpepper_noise(mask_image)
            mask_image = self.jpeg_compress(mask_image, self.jpeg)

            mask_image = transforms.ToTensor()(mask_image)
            image = torch.cat([image, mask_image], dim=0)

        
        return {"image":image, "label":label}
    
#添加椒盐噪声
class AddSaltPepperNoise(object):

    def __init__(self, density=0.1, p=0.0):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img

#添加Gaussian噪声
class AddGaussianNoise(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, p=0.0, mean=0.0, variance=5.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
