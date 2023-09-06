# Diffusion-Generated Fake Face Detection by Exploring Wavelet Domain Forgery Clues
This is the official repository of "Diffusion-Generated Fake Face Detection by Exploring Wavelet Domain Forgery Clues" (WCSP 2023)

## Dependencies
* Pytorch ≥ 1.10 with CUDA ≥ 11.3
* tensorboard
* opencv-python
* timm
* pyyaml
* tqdm

## Dataset and Pretrained Models
[Dropbox](https://www.dropbox.com/scl/fo/yayfd8si7uo8nk9u5bpl3/h?rlkey=2spxb2hbrp7m63rp7uwp805rj&dl=0)
[北航云盘](https://bhpan.buaa.edu.cn/link/AA7A99B20D16FB4C9EA3033D20E811E0A7) (提取码：wcsp)
* Dataset: Please unzip the downloaded datasets and put them in one folder, for example:
```
- /mnt/data/diffusion_detection
    - celeba_hq_256
        - test
        - train
        - val
    - wavediff_celeba256
        - ...
    - ddpm_celeba256_1000
        - ...
```
* Weights: put them in `./weights` folder.

## Start
* Training 
```
python -W ignore main.py --folder /mnt/data/diffusion_detection --cuda 0 --batch-size 64 --save --mask True --fake ddpm_celeba256_1000 --weights ours_ddpm.pth.tar --config config/test.yaml
```
* Testing
```
python run_robust.py
```
parameters to modify:
* `--fake` To modify fake dataset;
* `--cuda` To modify gpu id e.g. `--cuda 0` for one gpu, `--cuda 0 1` for two gpus;
* `--weights` To modify the saving weights file name during training and testing.
 
