import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.optim.optimizer import Optimizer, required

import datasets
from utils import AverageMeter, Logger
from center_loss import CenterLoss

parser = argparse.ArgumentParser("Logistic Regression Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--update_way', type=int, default=3)
'''
    update_way=1 : pure SGD
    update_way=2 : regularization + SGD, meanwhile reg means Regularization factor
    update_way=3 : add a random noise term after the gradient as the new gradient
'''
parser.add_argument('--reg', type=int, default=1) # Regularization factor
parser.add_argument('--weight_reg', type=int, default=0.001) # Regularization factor
# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log/LR')
args = parser.parse_args()

# LR model define
class LR(nn.Module):
    def __init__(self, input_dim=None, output_dim=None):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim) # bias = true by default

    def forward(self, x):
        bt = x.shape[0]
        x = x.view(bt, -1)
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out

# optimization redefine
class SGD_Plus(Optimizer):
    '''
    update=1 : pure SGD
    update=2 : regularization + SGD, meanwhile `reg` means Regularization factor, `weight_reg` means lambda
    update=3 : add a random noise term after the gradient as the new gradient, `weight_reg` means delta
    '''
    def __init__(self, params, lr=required, update=1, reg=1, weight_reg=0.1):
        defaults = dict(lr=lr, update=update, reg=reg, weight_reg=weight_reg)
        super(SGD_Plus, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            update = group['update']
            reg = group['reg']
            weight_reg = group['weight_reg']

            for p in group['params']: # p:torch.size([10,784])
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if update == 1:
                    p.data.add_(-group['lr'], d_p)
                elif update == 2:
                    if reg == 1: 
                        d_p.add_(weight_reg, torch.sign(p.data))  # L1正则导数
                    if reg == 2:
                        d_p.add_(weight_reg, 2*p.data) # L2正则导数
                    p.data.add_(-group['lr'], d_p)
                elif update == 3:
                    p.data.add_(-group['lr'], d_p)
                    p.data.add_(weight_reg, torch.randn((p.data.shape)).cuda())
        return loss


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers,
    )

    trainloader, testloader = dataset.trainloader, dataset.testloader

    print("Creating model: Logistic Regression")
    model = LR(input_dim=28*28, output_dim=dataset.num_classes)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD_Plus(model.parameters(), lr=args.lr_model, update=args.update_way,
                        reg=args.reg, weight_reg=args.weight_reg)
    
    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion, optimizer, trainloader, use_gpu, dataset.num_classes, epoch)

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            acc, err = test(model, testloader, use_gpu, dataset.num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    if args.update_way == 1:
        torch.save(model.state_dict(), '{}_{}_{}.pth'.format(args.update_way, args.lr_model, args.max_epoch))
    elif args.update_way == 2:
        torch.save(model.state_dict(), '{}_{}_{}_{}_{}.pth'.format(args.update_way, args.lr_model, args.max_epoch, args.reg, args.weight_reg))
    elif args.update_way == 3:
        torch.save(model.state_dict(), '{}_{}_{}_{}.pth'.format(args.update_way, args.lr_model, args.max_epoch, args.weight_reg))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, criterion, optimizer, trainloader, use_gpu, num_classes, epoch):
    model.train()
    losses = AverageMeter()

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    main()






