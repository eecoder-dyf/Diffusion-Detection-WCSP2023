import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ['TORCH_HOME'] = './model/pretrained_weights'

from torchvision import models
from model.ours import Network
from utils import *
from train import train_epoch, val_epoch
from dataset import FakeDataset
# python -W ignore main.py --folder ~/database/diffusion_detect --cuda 0 --batch-size 64 --save --mask True --fake ldm_celeba256_200 --weights ours_ldm.pth.tar --config config/test.yaml

def main(argv):
    args = parse_args(argv)

    if args.test:
        test_dataset = FakeDataset(folder=args.folder, real_data=args.real, fake_data=args.fake, normalize=True,
                                   phase='test', mask=args.mask, test_config=args.config)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                     pin_memory=True)
    else:
        train_dataset = FakeDataset(folder=args.folder, real_data=args.real, fake_data=args.fake, normalize=True, phase='train', mask=args.mask, test_config=args.config)
        val_dataset = FakeDataset(folder=args.folder, real_data=args.real, fake_data=args.fake, normalize=True, phase='val', mask=args.mask, test_config=args.config)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                    pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    device_ids = list(args.cuda)
    if device == 'cuda':
        torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
    print('temp gpu device number:')
    print(torch.cuda.current_device())

    net = Network(models.resnet18(pretrained=True), mask=args.mask)
    # print(net)
    # get_params_flops(tensor=torch.randn(1,3,256,256), network=net)
    # raise ValueError ("stop")
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda(device=device_ids[0])



    if args.pretrained is not None and os.path.exists(args.pretrained):
        pretrained_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage)['state_dict']
        # for name, para in net.named_parameters():
        #     print(name)
        # for k, v in pretrained_dict.items():
        #     print(k)
        model_dict = net.module.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_dict)
        net.module.load_state_dict(model_dict)
        # net.module = freeze_model(model=net.module, to_freeze_dict=pretrained_dict)
        print("load pretrained model ok")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, weight_decay=1e-5)
    get_params(net)
    weight_root = os.path.join("weights", args.weights)

    if os.path.exists(weight_root):
        model = torch.load(weight_root, map_location=lambda storage, loc: storage)
        net.module.load_state_dict(model['state_dict'], strict=True)
        if not args.test:
            optimizer.load_state_dict(model['optimizer'])
        epoch_now = model['epoch']
        print(f'load model self completion ok, weight file is {weight_root}')
    else:
        epoch_now = -1
        print('train from none')

    writer = SummaryWriter(f'./log/ours/{args.fake}')
    # input_x = torch.randn(1, 6, 256, 256)
    # writer.add_graph(net, input_x)

    if args.test:
        loss_test = val_epoch(epoch_now, test_dataloader, net, criterion, args.tsne)
        return loss_test
    
    best_loss = 1e10
    loss = val_epoch(epoch_now, val_dataloader, net, criterion)
    if loss < best_loss:
        best_loss = loss


    for epoch in range(epoch_now + 1, args.epoch + epoch_now + 1):
        train_loss = train_epoch(epoch, train_dataloader, net, criterion, optimizer)
        loss = val_epoch(epoch, val_dataloader, net, criterion)
        writer.add_scalar('train_loss', train_loss, global_step=epoch, walltime=None)
        writer.add_scalar('val_loss', loss, global_step=epoch, walltime=None)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        print(f'is_best: {is_best:}')
        if args.save:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': net.module.state_dict(),
                    'loss': loss,
                    'optimizer': optimizer.state_dict(),
                }, True, filename=weight_root) # 改成了True,每个都保存，注意改回来


if __name__ == '__main__':
    main(sys.argv[1:])