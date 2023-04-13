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
from models.ST_Former import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
from distutils.dir_util import copy_tree
import itertools
from dataloader.dataset_AFEW import train_data_loader, test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/' + time_str + 'log.txt'
log_curve_path = './log/' + time_str + 'log.png'
checkpoint_path = './checkpoint/' + time_str + 'model.pth'
best_checkpoint_path = './checkpoint/' + 'model_best.pth'
#pretrained_checkpoint_path = './checkpoint/model_pretrained_on_DFEW_fd1.pth'
pretrained_checkpoint_path = r'./checkpoint/model_best.pth'
# pretrained_checkpoint_path = None

labels = ['Happy','Sad','Neutral','Angry','Surprise','Disgust','Fear']
labels_3 = ['Positive','Neutral','Negative']
# labels = ['Negative','Neutral','Positive']
# labels = ['Negative','Neutral']

def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        print(n)
        nlabel = '{0}'.format(labels[k])
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0:.2f}'.format(con_mat[i, j] * 100 / con_mat[i].sum()), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




def main():
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))

    # create model and load pre_trained parameters
    model = GenerateModel()

    # model2D = CAERSNet()
    # # path_2D = 'D:/project/CAERS-master/CAER-master/CAER/saved/models/CAERS_Session/224_aug3/model_best.pth'
    # # checkpoint = torch.load(path_2D)
    # # state_dict = checkpoint['state_dict']
    # # model2D.load_state_dict(state_dict)
    #
    # model = CAERSNet_video_Transf(model2D, num_frames=16)
    #############################################################################
    model = torch.nn.DataParallel(model).cuda()
    if pretrained_checkpoint_path != None:
        pre_trained_dict = torch.load(pretrained_checkpoint_path)['state_dict']
        model.load_state_dict(pre_trained_dict,strict=True)
        print('load pretrained model')
    else:
        print('training from scratch')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    # Data loading code
    #train_data = train_data_loader()
    test_data = test_data_loader()

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             drop_last = True)


    # evaluate on validation set

    val_acc, val_los = validate(val_loader, model, criterion, args)






def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top = []
    for i in range(11):
        top.append(AverageMeter('Accuracy', ':6.3f'))

    for i in range(11):
        top.append(AverageMeter('Accuracy', ':6.3f'))

    top.append(AverageMeter('Accuracy', ':6.3f'))

    progress = ProgressMeter(len(val_loader),
                             [losses, top[4]],
                             prefix='Test: ')

    model_2D = GenerateModel_2D()
    model_2D = torch.nn.DataParallel(model_2D).cuda()
    if pretrained_checkpoint_path != None:
        pre_trained_dict = torch.load(pretrained_checkpoint_path)['state_dict']
        model_2D.load_state_dict(pre_trained_dict, strict=False)


    # switch to evaluate mode
    model.eval()
    model.eval()

    true_labels_3 = []
    pred_labels_3 = []
    true_labels = []
    pred_labels = []

    cnt = 0

    start_time = time.time()

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

            # PG, PL1, PL2, P2D, x ,xs = model(images)
            PG, PL1, PL2, P2D, x, xs, out_sig_PG, out_sig_PL1, out_sig_PL2 = model(images)

            with torch.no_grad():
                _, _, _, P2D_2D = model_2D(images)
                
            # with open("Decision_log.txt", "a") as file:
            #     decision = "(%d - %s) : "%(i, labels[target])
            #     for de in P2D_2D.max(dim=1).indices:
            #         decision += "%s, "%labels[de]
            #     file.write('%s\n' % decision)
            #     file.close()
            #
            # with open("score_log.txt", "a") as file:
            #     score = "(%d - %s) : "%(i, labels[target])
            #     for de in torch.nn.Softmax()(P2D_2D)[:,target[0]]:
            #         score += "%.2f, "%de
            #     file.write('%s\n' % score)
            #     file.close()

            # P2D = torch.log_softmax(P2D, dim=1).view(1, 16, -1)
            # P2D = torch.mean(P2D, dim=1)

            alpha = 0.7

            PG1 = []
            PG1.append((PG[:, :1] + PG[:, 4:5]))
            PG1.append(PG[:, 2:3])
            PG1.append((PG[:, 1:2] + PG[:, 3:4] + PG[:, 5:6] + PG[:, 6:]))
            PG1 = torch.cat(PG1, dim=1)

            loss_P1 = alpha * nn.NLLLoss()(PL1, target_3) + (1 - alpha) * nn.NLLLoss()(PG1, target_3)  # semantic level
            loss_P2 = alpha * nn.NLLLoss()(PL2, target) + (1 - alpha) * nn.NLLLoss()(PG, target)  # affective level

            loss = loss_P1 + loss_P2

            # measure accuracy and record loss
            #### 3class
            out_3 = []
            beta = 0
            for j in range(11):
                out_3.append((beta * PG1 + (1 - beta) * PL1))
                beta += 0.1

            acc3 = []
            for outr in out_3:
                acc3.append(accuracy(outr, target_3, topk=(1, 2))[0])

            for j in range(11):
                top[j].update(acc3[j][0], images.size(0))

            P3 = PG.clone()
            for t in range(len(P3[0])):
                if t in [0, 4]:  # Positive (happy surprise)
                    P3[0][t] = PL1[0][0]
                elif t == 2:  # Neutral
                    P3[0][t] = PL1[0][1]
                else:  # Negative (Sad, angry, disgust, fear)
                    P3[0][t] = PL1[0][2]


            P2D = torch.log_softmax(P2D,dim=1).view(1,16,-1)
            P2D = torch.mean(P2D, dim=1)
            out_r = []
            beta = 0
            for j in range(11):
                # out_r.append((beta * PG + (1 - beta) * PL2))
                out_r.append((beta * PG + (1 - beta) * PL2) )
                beta += 0.1


            acc = []
            for outr in out_r:
                acc.append(accuracy(outr, target, topk=(1, 2))[0])


            losses.update(loss.item(), images.size(0))

            for j in range(11):
                top[11+j].update(acc[j][0], images.size(0))

            ###
            PGtemp = []
            PGtemp.append((out_r[3][:, :1] + out_r[3][:, 4:5])/2)
            PGtemp.append(out_r[3][:, 2:3])
            PGtemp.append((out_r[3][:, 1:2] + out_r[3][:, 3:4] + out_r[3][:, 5:6] + out_r[3][:, 6:])/4)
            PGtemp = torch.cat(PGtemp, dim=1)
            out_temp = accuracy(PGtemp + 5* PL1, target_3, topk=(1, 2))[0]
            top[22].update(out_temp[0], images.size(0))
            # confusion matrix
            pred_sum = out_r[3]
            pred_sum_3 = PGtemp

            if pred_sum.argmax() in [0, 4]:  # Positive (happy surprise)
                pred_sum_3 = torch.tensor([[1,0,0]]).cuda()
            elif pred_sum.argmax() == 2:  # Neutral
                pred_sum_3 = torch.tensor([[0,1,0]]).cuda()
            else:  # Negative (Sad, angry, disgust, fear)
                pred_sum_3 = torch.tensor([[0,0,1]]).cuda()


            pred_labels_3.append(pred_sum_3.cpu().detach().numpy())
            true_labels_3.append(target_3.cpu().detach().numpy())

            pred_labels.append(pred_sum.cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())
            

            cnt += 1
            if i % args.print_freq == 0:
                progress.display(i)

        epoch_time = time.time() - start_time
        print('An epoch time: {:.1f}s'.format(epoch_time))
        print('Second per a sample {:.3f}s'.format(epoch_time/cnt))
        # TODO: this should also be done with the ProgressMeter
        for j in range(23):
            print('Current Accuracy: {top1.avg:.3f}'.format(top1=top[j]))

        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top2))
        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top3))

        

        pred_labels = np.argmax(np.array(pred_labels), axis=2)
        pred_labels_3 = np.argmax(np.array(pred_labels_3), axis=2)

        plot_confusion_matrix(confusion_matrix(true_labels, pred_labels), labels=labels,
                              normalize=True)

        plot_confusion_matrix(confusion_matrix(true_labels_3, pred_labels_3), labels=labels_3,
                              normalize=True)
        
        
    return top[4].avg, losses.avg


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        #shutil.copyfile(checkpoint_path, best_checkpoint_path)
        torch.save(state, best_checkpoint_path)



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
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
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
