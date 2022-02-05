import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import dataset
import OPA_supervised_binomial
import sklearn.metrics
from scipy.stats import mode
import os.path as osp



parser = argparse.ArgumentParser(description='Temporal Alignment Prediction for Supervised Representation Learning and Few-Shot Sequence Classification')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-savedir', metavar='MIR',
                    help='path to save models')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr') #0.03
parser.add_argument('--schedule', default=[160, 240, 320, 400], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)') #[120, 160, 240, 300]
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# TAP specific configs:
parser.add_argument('--inputdim', default=100, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--middledim', default=1024, type=int,
                    help='hidden nodes in encoder and predictor')
parser.add_argument('--outputdim', default=128, type=float,
                    help='feature dimension of transformed sequences')
parser.add_argument('--lam', default=1, type=float,
                    help='weight of relative temporal positions')
parser.add_argument('--lam1', default=1, type=float,
                    help='weight of dual meta-distance')
parser.add_argument('--lam2', default=0.5, type=float,
                    help='weight of dual alignment')
parser.add_argument('--normalized', action='store_true',
                    help='add a normalization layer to encoder and predictor')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False

setup_seed(892101)

def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = OPA_supervised_binomial.BlurContrastiveModelPair(input_dim=args.inputdim)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = OPA.BlurContrastiveLossPair()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    datasetall = dataset.SequenceDataset(args.data) #, test_dataset
    #train_dataset = datasetall.traindata

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        datasetall.traindata, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=_init_fn)
    
    traintest_loader = torch.utils.data.DataLoader(
        datasetall.traindata, batch_size=datasetall.traindata.__len__(), shuffle=False,
        num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    test_loader = torch.utils.data.DataLoader(
        datasetall.testdata, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    #os.mkdir(args.save_dir)
    all_epoch_time = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        epochstart = time.time()
        train(train_loader, model, optimizer, epoch, args)
        epochend = time.time()
        all_epoch_time.append(epochend-epochstart)

        if epoch%10==0 and epoch>40:  #
            acc1 = validate(test_loader, traintest_loader, model, args)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    #'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=osp.join(args.savedir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))
                #fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar')
    print(all_epoch_time)
    print(sum(all_epoch_time))


def train(train_loader, model, optimizer, epoch, args):  #, criterion
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, labels, lens) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)       

        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            lens = lens.cuda(args.gpu, non_blocking=True)

        # compute output
        loss = model(inputs, lens, labels)
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, traintest_loader, model, args):
    test_batch_time = AverageMeter('Time', ':6.3f')
    #losses = AverageMeter('Loss', ':.4e')
    map1 = AverageMeter('Map@1', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top7 = AverageMeter('Acc@7', ':6.2f')
    top15 = AverageMeter('Acc@15', ':6.2f')
    top30 = AverageMeter('Acc@30', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [test_batch_time, map1, top1, top3, top5, top7, top15, top30],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for j, (trinputs, trlabels, trlens) in enumerate(traintest_loader):
            #print(j)
            if args.gpu is not None:
                trinputs = trinputs.cuda(args.gpu, non_blocking=True)
                labels2 = trlabels.cuda(args.gpu, non_blocking=True)
                lens2 = trlens.cuda(args.gpu, non_blocking=True)
            #seq2 = trinputs  #model.encoder(trinputs)
            seq2 = model.encoder(trinputs)
            R2 = model.getlen(seq2,lens2) #.cuda(args.gpu, non_blocking=True)

            trnum = seq2.size(0)
            trlabels = trlabels.numpy().reshape(-1) 
            #print(trlabels)
            for i, (inputs, labels, lens) in enumerate(val_loader):
                if args.gpu is not None:
                    inputs = inputs.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)
                    lens = lens.cuda(args.gpu, non_blocking=True)
                # compute output
                #seq1 = inputs  #model.encoder(inputs)
                seq1 = model.encoder(inputs)
                seq1 = seq1[:,0:lens[0],:]
                R1 = model.getlen(seq1,lens) #.cuda(args.gpu, non_blocking=True)
                # print(R1.device)
                # print(seq1.device)
                tenum = seq1.size(0)
                assert(tenum==1)
                dismat = np.zeros(trnum)
                for trcount in range(trnum):
                    seqtr = seq2[trcount,0:lens2[trcount],:].unsqueeze(0)
                    Rtr = R2[trcount,0:lens2[trcount]].unsqueeze(0)
                    # print(Rtr.device)
                    # print(seqtr.device)
                    _,dis = model.alignment(seq1,seqtr,R1,Rtr)
                    #dis = dis.view(-1)
                    #print(dis)
                    dismat[trcount] = dis[0].item()
                    
                labels = labels[0].item()   
                #print(labels)        
                knn_idx_full = dismat.argsort()
                #print(knn_idx_full)
                acc = np.zeros(6)
                count = 0
                for k in [1, 3, 5, 7, 15, 30]:
                    knn_idx = knn_idx_full[0:k]  #[:, :k]
                    knn_labels = trlabels[knn_idx]
                    #print(knn_labels)
                    mode_data = mode(knn_labels)  #, axis=1
                    mode_label = mode_data[0]
                    #print(mode_label[0])
                    #print(mode_data)
                    if mode_label[0]==labels:
                        acc[count] = acc[count]+100
                    count = count + 1
                
                tru_label = np.where(trlabels==labels,1,0).reshape(-1)
                #print(tru_label)
                ap = sklearn.metrics.average_precision_score(tru_label,-dismat.reshape(-1))

                # measure accuracy and record loss
                map1.update(ap, inputs.size(0))
                top1.update(acc[0], inputs.size(0))
                top3.update(acc[1], inputs.size(0))
                top5.update(acc[2], inputs.size(0))
                top7.update(acc[3], inputs.size(0))
                top15.update(acc[4], inputs.size(0))
                top30.update(acc[5], inputs.size(0))

                # measure elapsed time
                test_batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Map@1 {map1.avg:.3f} Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f} Acc@7 {top7.avg:.3f} Acc@15 {top15.avg:.3f} Acc@30 {top30.avg:.3f}'
              .format(map1=map1, top1=top1, top3=top3, top5=top5, top7=top7, top15=top15, top30=top30))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
