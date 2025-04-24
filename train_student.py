"""
the general training framework
"""

from __future__ import print_function
import random
import sys
import numpy as np
import os
import argparse
import socket
import time
from timm.scheduler.cosine_lr import CosineLRScheduler
import logging
# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from models_ import model_dict
from models_.util import Embed, ConvReg, LinearEmbed
from models_.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.miniImage import get_miniImagenet_dataloader,get_dataloader_sample


from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss
from termcolor import colored
from helper.util import  accuracy, AverageMeter,adjust_learning_rate
from helper.pretrain import init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    path_file = os.path.join(output_dir, f'{name}log_rank{dist_rank}.txt')
    if os.path.isfile(path_file):
        os.remove(path_file)
    file_handler = logging.FileHandler(path_file,mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def train(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)
            index = index.to(device)
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.to(device)

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        # for i in range(len(feat_s)):
        #     print(feat_s[i].shape)
        with torch.no_grad():
            # feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            logit_t = model_t(input, is_feat=False, preact=preact)
            # feat_t = [f.detach() for f in feat_t]
            # print('---')
            # for i in range(len(feat_t)):
            #     print(feat_t[i].shape)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity4':
            g_s = [feat_s[-4]]
            g_t = [feat_t[-4]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step_update(epoch * len(train_loader) + idx)

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','miniImage'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='mobile_vit_tiny_likevit',
                       choices=['MobileNetV2','MobileNetV3' 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
                                  'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
                                   'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best',
                                   'mobile_vit_tiny_likevitpp','mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
                                   'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
                                   'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
                                   'mobile_vit_tiny_dilatedblock_5_noweight','mobilecontainer','repvit_m0_6','repvit_m0_6_infiuni','RepInfiniteUniVit_initial','repvit_m0_6_uni_ours',
                                   'mobile_half_1_1_1','mobile_half_1_1_2','mobile_half_1_1_3','mobile_half_1_1_4','mobile_half_1_2_1','mobile_half_1_2_2','mobile_half_1_2_3','mobile_half_1_2_4','mobile_half_1_3_1','mobile_half_1_1_5',
                                   'mobile_half_2_1_1','mobile_half_2_1_1_1','mobile_half_2_1_2','mobile_half_2_1_3','mobile_half_2_1_4','mobile_half_2_1_5','mobile_half_2_1_6','mobile_half_3_1_1',
                                   'mobile_half_4_1_1','mobile_half_4_1_2','mobile_half_4_1_3','mobile_half_4_1_4','mobile_half_4_1_5','mobile_half_4_1_6','mobile_half_4_1_7','mobile_half_5_1_1','mobile_half_percent',
                                   'mobile_half_1_1_7','mobile_half_1_1_8','mobile_half_1_1_9','mobile_half_1_1_1_1','mobile_half_1_1_10','mobile_half_1_1_11','mobile_half_1_1_12','mobile_half_4_1_2_1','mobile_half_4_1_2_2','mobile_half_4_1_3_1','mobile_half_4_1_3_3','mobile_half_4_1_3_2','mobile_half_1_1_12_1',
                                   'mobile_half_1_1_8_1','mobile_half_1_1_8_2','mobile_half_6_1_1_1','mobile_half_6_1_1_2','mobile_half_6_1_2_1','mobile_half_1_1_8_3','mobile_half_6_1_1_2_1','mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1',
                                   'mobile_half_1_2_2_1','mobile_half_1_2_1_1','mobile_half_7_1_1','mobile_half_7_1_2','mobile_half_7_1_1_1','mobile_half_7_1_2_1','mobile_half_1_1_1_2','mobile_half_5_1_1_1','mobile_half_5_1_1_2','mobile_half_1_1_8_4','mobile_half_1_1_8_5','mobile_half_1_1_8_6','mobile_half_1_1_8_7',
                                   'mobilemetanet','mobilemetanet_1','mobilemetanet_2','mobile_half_3_1_2','RepTinynet','RepTinynet1','RepTinynet2','RepTinynet3','RepTinynet4','RepTinynet5','RepTinynet6','RepTinynet7','RepTinynet8','RepTinynet9','RepTinynet10','RepTinynet11','RepTinynet12','RepTinynet13','RepTinynet14','RepTinynet15','mcunetlike','RepTinynet16','RepTinynet17',
                                   'mobile_half_5_1_1_3','mobile_half_class','mobile_half_class_2','mobile_half_class_3','RepTinynet18','RepTinynet19','RepTinynet20','RepTinynet21','RepTinynet22','RepTinynet23','mobile_half_5_1_1_4','RepTinynet24','RepTinynet25','MobilenetV1','MobilenetV3_m','XceptionNet','Mobilenetxtnet','squeezenet','MCUnetv1'])
    parser.add_argument('--path_t', type=str, default='/root/distill/path/teacher_model/mobile_vit_small_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_best.pth', help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    parser.add_argument('--arch_name', type=str, default='kd_mobile_vit_tiny_likevit',
                        help='log name')
    parser.add_argument('--OUTPUT', type=str, default='./log',
                        help='log output path')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
                                  'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
                                   'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best',
                                   'mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
                                   'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
                                   'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
                                   'mobile_vit_tiny_dilatedblock_5_noweight','mobile_vit_tiny_dilatedblock_5_5_1']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('wsl'):
        opt.model_path = './path/student_model'
        opt.tb_path = './path/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path+opt.distill, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    print(segments)
    if segments[0] == 'mobile':
        return segments[0] + '_' + segments[1] + '_' + segments[2]+ '_' + segments[3]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    print(model_t)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    seed = 970203
    g = torch.Generator()
    g.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)     # 当前GPU
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    best_acc = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parse_option()
    loggering = create_logger(output_dir=opt.OUTPUT, dist_rank=0, name=f"{opt.arch_name}")
    # tensorboard logger
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    elif opt.dataset == 'miniImage':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_dataloader_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader,n_data = get_miniImagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers,is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 224, 224)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'mask':
        criterion_kd = MaskLoss()

    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, loggering, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, loggering, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, loggering, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    # total_steps = opt.epochs*len(train_loader)
    # warmup_steps  = int(len(train_loader))
    # lr_scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=total_steps,
    #         lr_min=0.0001,
    #         warmup_lr_init=0.0,
    #         warmup_t=warmup_steps,
    #         cycle_limit=1,
    #         t_in_epochs=False,
    #     )

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.to(device)
        criterion_list.to(device)
        # cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    acc_arr = []
    acc5_arr = []
    loss_arr = []
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        if epoch==1:
            if opt.distill in ['crd']:
                for input,_,_,_ in (train_loader):
                    model_s.eval()
                    img = input[0]
                    img = torch.unsqueeze(img, dim=0)
                    # predict class
                    flops = FlopCountAnalysis(model_s,img.to(device))
                    params = parameter_count_table(model_s)
                    print(flops.total())
                    print(params)
                    loggering.info(flops.total())
                    loggering.info(params)
                    model_s.train()
                    break
            else:
                for input,_,_ in (train_loader):
                    model_s.eval()
                    img = input[0]
                    img = torch.unsqueeze(img, dim=0)
                    # predict class
                    flops = FlopCountAnalysis(model_s,img.to(device))
                    params = parameter_count_table(model_s)
                    print(flops.total())
                    print(params)
                    loggering.info(flops.total())
                    loggering.info(params)
                    model_s.train()
                    break

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        
        acc_arr.append(float(test_acc))
        acc5_arr.append(float(test_acc_top5))
        loss_arr.append(float(test_loss))
        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_loss', test_loss, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        loggering.info(f"test_acc{test_acc} epoch{epoch} top5 {test_acc_top5} loss{test_loss}")
        loggering.info(f"acc1{acc_arr}")
        loggering.info(f"acc5{acc5_arr}")
        loggering.info(f"loss{loss_arr}")

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
