import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.sampler import WeightedRandomSampler
from models.ST_Former import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
from distutils.dir_util import copy_tree
from dataloader.dataset_AFEW import train_data_loader, test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/' + time_str + 'log.txt'
log_curve_path = './log/' + time_str + 'log.png'
checkpoint_path = './checkpoint/' + time_str + 'model.pth'
best_checkpoint_path = './checkpoint/' + 'model_best.pth'
best_model_checkpoint_path = './checkpoint/' + 'only_model_best.pth'

pretrained_checkpoint_path = r'./checkpoint/2D_pretrain.pth'


# pretrained_checkpoint_path = None


def main():
    best_acc = 0
    max_beta = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))

    # create model and load pre_trained parameters
    model = GenerateModel()
    model_2D = GenerateModel_2D()


    #############################################################################
    model = torch.nn.DataParallel(model).cuda()
    model_2D = torch.nn.DataParallel(model_2D).cuda()
    Pram = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of parameters : %d ' % Pram)
    if pretrained_checkpoint_path != None:
        pre_trained_dict = torch.load(pretrained_checkpoint_path)['state_dict']
        model.load_state_dict(pre_trained_dict, strict=True)
        model_2D.load_state_dict(pre_trained_dict, strict=False)
        print('load pretrained model')
    else:
        print('training from scratch')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # 80

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader()
    test_data = test_data_loader()

    weights = train_data.make_weights_for_balanced_classes()
    # sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               # sampler=sampler,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,  # 1,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             drop_last=False)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        # train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)
        train_acc, train_los = train_keyframed(train_loader, model, model_2D, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc and save checkpoint
        avgs = []
        for i in range(11):
            avgs.append(val_acc[i].avg)

        val_acc = np.max(avgs)

        is_best = val_acc > best_acc
        if is_best:
            max_beta = np.argmax(avgs)
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()) + 'beta = %d' % max_beta)
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    sample_rate = np.zeros(7)
    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        target_3 = target.clone()
        for t in range(len(target)):
            if target[t] in [0, 4]:  # Positive (happy surprise)
                target_3[t] = 0
            elif target[t] == 2:  # Neutral
                target_3[t] = 1
            else:  # Negative (Sad, angry, disgust, fear)
                target_3[t] = 2

        target_t = target.clone()
        target_2D = []
        for b in range(16):
            target_2D.append(target_t.view(args.batch_size, -1))
        target_2D = torch.cat(target_2D, dim=1)
        target_2D = target_2D.view(16 * args.batch_size)

        for s in target:
            sample_rate[s] += 1

        # compute output
        PG, PL1, PL2, P2D = model(images)

        alpha = 0.7

        PG1 = []
        PG1.append((PG[:, :1] + PG[:, 4:5]) / 2)
        PG1.append(PG[:, 2:3])
        PG1.append((PG[:, 1:2] + PG[:, 3:4] + PG[:, 5:6] + PG[:, 6:]) / 4)
        PG1 = torch.cat(PG1, dim=1)

        loss_P1 = alpha * nn.NLLLoss()(PL1, target_3) + (1 - alpha) * nn.NLLLoss()(PG1, target_3)  # semantic level
        loss_P2 = alpha * nn.NLLLoss()(PL2, target) + (1 - alpha) * nn.NLLLoss()(PG, target)  # affective level
        loss_P2D = criterion(P2D, target_2D)

        loss = 0.6 * loss_P1 + 1.4 * loss_P2  # + 0.1*loss_P2D

        # measure accuracy and record loss
        acc1, _ = accuracy(PG, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    sample_rate = sample_rate / np.sum(sample_rate)
    print(sample_rate)

    return top1.avg, losses.avg


def train_keyframed(train_loader, model, model_2D, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model_2D.eval()
    sample_rate = np.zeros(7)
    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        target_3 = target.clone()
        for t in range(len(target)):
            if target[t] in [0, 4]:  # Positive (happy surprise)
                target_3[t] = 0
            elif target[t] == 2:  # Neutral
                target_3[t] = 1
            else:  # Negative (Sad, angry, disgust, fear)
                target_3[t] = 2

        target_t = target.clone()
        target_2D = []
        for b in range(16):
            target_2D.append(target_t.view(args.batch_size, -1))
        target_2D = torch.cat(target_2D, dim=1)
        target_2D = target_2D.view(16 * args.batch_size)

        for s in target:
            sample_rate[s] += 1

        # compute output
        PG, PL1, PL2, P2D, x, xs, out_sig_PG, out_sig_PL1, out_sig_PL2 = model(images)


        with torch.no_grad():
            _, _, _, P2D_2D = model_2D(images)

        # torch.sign(torch.abs(P2D_2D * F.one_hot(target_2D, num_classes=7)))
        # (P2D_2D * F.one_hot(target_2D, num_classes=7)).view(32, 16, -1).sum(dim=2,keepdim=True)
        # ((P2D_2D * F.one_hot(target_2D, num_classes=7)).view(32, 16, -1).sum(dim=2,keepdim=True)*xs[0].view(32,16,-1)).sum(dim=1)
        # frame_scores = (P2D_2D * F.one_hot(target_2D, num_classes=7)).view(32, 16, -1).max(dim=2,keepdim=True)[0].detach() #### hard label
        frame_scores = (P2D_2D * F.one_hot(target_2D, num_classes=7)).view(32, 16, -1).sum(dim=2, keepdim=True).detach()

        weighted_frame_sum_0 = (frame_scores * xs[0].view(args.batch_size, 16, -1)).sum(dim=1)
        weighted_frame_sum_1 = (frame_scores * xs[1].view(args.batch_size, 16, -1)).sum(dim=1)
        weighted_frame_sum_3 = (frame_scores * xs[3].view(args.batch_size, 16, -1)).sum(dim=1)

        L2_loss = torch.nn.L1Loss()

        Semantic_loss_0 = L2_loss(x[0], weighted_frame_sum_0) / x[0].shape[1]
        Semantic_loss_1 = L2_loss(x[1], weighted_frame_sum_1) / x[1].shape[1]
        Semantic_loss_3 = L2_loss(x[3], weighted_frame_sum_3) / x[3].shape[1]

        loss_S2T = (1 * Semantic_loss_0 + 1 * Semantic_loss_1 + 1 * Semantic_loss_3) * 3 # Lta

        alpha = 0.7

        PG1 = []
        PG1.append((PG[:, :1] + PG[:, 4:5]) / 2)
        PG1.append(PG[:, 2:3])
        PG1.append((PG[:, 1:2] + PG[:, 3:4] + PG[:, 5:6] + PG[:, 6:]) / 4)
        PG1 = torch.cat(PG1, dim=1)

        E = nn.Softmax()(P2D_2D.view(32, 16, -1).sum(dim=1)).detach()
        Energy1 = []
        Energy1.append((E[:, :1] + E[:, 4:5]) / 2)
        Energy1.append(E[:, 2:3])
        Energy1.append((E[:, 1:2] + E[:, 3:4] + E[:, 5:6] + E[:, 6:]) / 4)
        Energy1 = torch.cat(Energy1, dim=1)

        loss_P1 = alpha * nn.NLLLoss()(PL1, target_3) + (1 - alpha) * nn.NLLLoss()(PG1, target_3)  # low affective level
        loss_P2 = alpha * nn.NLLLoss()(PL2, target) + (1 - alpha) * nn.NLLLoss()(PG, target)  # high affective level
        # loss_P2D = criterion(P2D, target_2D)

        # Laa
        loss_energy2 = L2_loss(nn.Sigmoid()(out_sig_PL2), nn.Softmax()(P2D_2D.view(32, 16, -1).sum(dim=1)).detach()) # out_sig_PL2
        loss_energy1 = L2_loss(nn.Sigmoid()(out_sig_PL1), Energy1) # out_sig_PL1

        loss_single = nn.NLLLoss()(PG, target)
        loss = 0.6 * loss_P1 + 1.4 * loss_P2 + 0.4 * loss_S2T + 0.7*loss_energy1 + 0.3*loss_energy2



        # measure accuracy and record loss
        acc1, _ = accuracy(PG, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    sample_rate = sample_rate / np.sum(sample_rate)
    print(sample_rate)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top = []
    for i in range(11):
        top.append(AverageMeter('Accuracy', ':6.3f'))
    progress = ProgressMeter(len(val_loader),
                             [losses, top[4]],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            target_3 = target.clone()
            for t in range(len(target)):
                if target[t] in [0, 4]:  # Positive (happy surprise)
                    target_3[t] = 0
                elif target[t] == 2:  # Neutral
                    target_3[t] = 1
                else:  # Negative (Sad, angry, disgust, fear)
                    target_3[t] = 2

            # compute output
            # output = model(images)

            PG, PL1, PL2, P2D, x, xs, out_sig_PG, out_sig_PL1, out_sig_PL2 = model(images)

            # P2D = torch.log_softmax(P2D, dim=1).view(args.batch_size,16,-1)
            # P2D = torch.mean(P2D, dim=1)

            alpha = 0.7

            PG1 = []
            PG1.append((PG[:, :1] + PG[:, 4:5]) / 2)
            PG1.append(PG[:, 2:3])
            PG1.append((PG[:, 1:2] + PG[:, 3:4] + PG[:, 5:6] + PG[:, 6:]) / 4)
            PG1 = torch.cat(PG1, dim=1)

            loss_P1 = alpha * nn.NLLLoss()(PL1, target_3) + (1 - alpha) * nn.NLLLoss()(PG1, target_3)  # semantic level
            loss_P2 = alpha * nn.NLLLoss()(PL2, target) + (1 - alpha) * nn.NLLLoss()(PG, target)  # affective level

            loss = loss_P1 + loss_P2

            # measure accuracy and record loss
            out_r = []
            beta = 0
            for j in range(11):
                out_r.append((beta * PG + (1 - beta) * PL2))
                beta += 0.1

            acc = []
            for outr in out_r:
                acc.append(accuracy(outr, target, topk=(1, 2))[0])

            losses.update(loss.item(), images.size(0))

            for j in range(11):
                top[j].update(acc[j][0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        for j in range(11):
            print('Current Accuracy: {top1.avg:.3f}'.format(top1=top[j]))
        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top2))
        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top3))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top[4]) + '\n')
    return top, losses.avg


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        # shutil.copyfile(checkpoint_path, best_checkpoint_path)
        torch.save(state, best_checkpoint_path)
        torch.save(state['state_dict'], best_model_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
