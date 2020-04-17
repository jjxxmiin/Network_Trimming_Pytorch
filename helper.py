import time
import pickle
import logging
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def train(model, train_loader, criterion, optimizer, epoch_log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    train_iter = len(train_loader)

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print(f'{epoch_log} \n'
              f'Iter: [{i}/{train_iter}] \n'
              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \n'
              f'Data {data_time.val:.3f} ({data_time.avg:.3f}) \n'
              f'Loss {losses.val:.4f} ({losses.avg:.4f}) \n'
              f'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'
              f'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \n')


def valid(model, valid_loader, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()

    valid_iter = len(valid_loader)

    for i, (input, target) in enumerate(valid_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print(f'Iter: [{i}/{valid_iter}]\n'
              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
              f'Loss {losses.val:.4f} ({losses.avg:.4f})\n'
              f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'
              f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n')

    print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} \n')

    return top1.avg, top5.avg


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
