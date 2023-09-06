import os
import glob
configs = glob.glob("config/*.yaml")
# print(configs)
models = {
          "ddpm":"ddpm_celeba256_1000",
        #   "ddim":"ddim_celeba256_200",
        #   "pndm":"pndm_celeba256_250",
        #   "ncsnpp":"ncsnpp_celeba256",
        #   "ldm":"ldm_celeba256_200",
          "wavediff":"wavediff_celeba256",
          }

folder = "./log/ours_robust_true"

for key in models.keys():
    os.makedirs(os.path.join(folder, key), exist_ok=True)
    for config in configs:
        directory, ext = os.path.splitext(config)
        configname = directory.split("/")[-1]
        logfile = os.path.join(folder, key, configname+".log")
        print(logfile)
        os.system(f"python -W ignore main.py --folder ~/database/diffusion_detect --cuda 0 --batch-size 64 --save --mask True --fake {models[key]} --weights ours_{key}.pth.tar --test --config {config} >> {logfile}")