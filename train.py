import os
import sys
import torch
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import AUROC
import numpy as np

def train_epoch(epoch, dataloader, model, criterion, optimizer):
    device = next(model.parameters()).device
    model.train()
    Loss = AverageMeter()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    for i, d in loop:
        image = d['image'].to(device)
        label = d['label'].to(device)
        result = model(image)
        optimizer.zero_grad()
        loss = criterion(result, label)
        loss.backward()
        optimizer.step()
        Loss.update(loss)
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=Loss.avg.item())
    return Loss.avg

def val_epoch(epoch, dataloader, model, criterion, tsne=False):
    device = next(model.parameters()).device
    model.eval()
    Loss = AverageMeter()
    Acc = AverageMeter()
    Pre = AverageMeter()
    Rec = AverageMeter()
    Auroc = AverageMeter()
    auroc = AUROC(task="binary")
    loop = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    npdict = {"wavefeat1":None, "wavefeat2":None, "wavefeat3":None, "feat4":None, "label":None}
    for i, d in loop:
        image = d['image'].to(device)
        label = d['label'].to(device)
        with torch.no_grad():
            if tsne:
                result, feats = model(image, tsne=True)
            else:
                result = model(image)
        if tsne:
            feats["label"] = label
            for key in feats.keys():
                tensor = feats[key]
                tensor = torch.flatten(input=tensor, start_dim=1, end_dim=-1)
                tensor = tensor.detach().cpu()
                ndarray = tensor.numpy()
                if i == 0:
                    npdict[key] = ndarray
                else:
                    npdict[key] = np.append(npdict[key], ndarray, axis=0)      

        loss = criterion(result, label)
        Loss.update(loss)
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=Loss.avg.item())
        predict = torch.max(result, dim=1)[1]
        for j in range(len(predict)):
            target = label[j, predict[j]]   # 预测正确就是1， 错误就是0
            acc = target.type(torch.float32)
            Acc.update(acc)
            auroc_cal = auroc(result.cpu(), label.cpu())
            Auroc.update(auroc_cal)
            if label[j,1] == 1: # 实际为正样本，计算Recall
                Rec.update(target.type(torch.float32))
            if predict[j] == 1: # 预测为正样本, 计算Precision 
                Pre.update(target.type(torch.float32))
    print(f"Epoch {epoch}, Prediction Accuracy: {Acc.avg:.5f}\tPrecision: {Pre.avg:.5f} ({Pre.sum}/{Pre.count})\tRecall: {Rec.avg:.5f} ({Rec.sum}/{Rec.count}), AUROC: {Auroc.avg:.5f}")
    if tsne:
        np.save('tsne_map/tsne.npy', npdict)
    return Loss.avg
