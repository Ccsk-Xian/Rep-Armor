# 普通模型训练
from __future__ import print_function

import os
import argparse
import socket
import time
import itertools
# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from timm.scheduler.cosine_lr import CosineLRScheduler
from model import model_dict
import sys
import numpy as np
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.miniImage import get_miniImagenet_dataloader
# import logging
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from helper.util import  accuracy, AverageMeter,adjust_learning_rate
# from helper.loops import train_vanilla as train, validate
from termcolor import colored
import timm

def validate(val_loader, model, criterion, opt,device):
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
            # loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    thr=1

    return top1.avg, top5.avg
       










def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--freeze-layers', type=bool, default=False)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    # dataset
    parser.add_argument('--model', type=str, default='mobile_vit_tiny_likevitpp',
                       choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2','MobileNetV3', 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
                                  'mobile_vit_small_init','mobile_vit_tiny_init','mobile_vit_xx_small_in7','mobile_vit_x_small_in7',
                                   'mobile_vit_small_in7','mobile_vit_tiny_in7','mobile_vit_tiny_likevit','mobile_vit_tiny_novit','mobile_vit_xx_small_best','mobile_vit_tiny_best','mobile_vit_tiny_likevitpp',
                                   'mobile_vit_tiny_novit_test1','mobile_vit_tiny_novit_test2','mobile_vit_tiny_novit_test3','mobile_vit_tiny_novit_test4',
                                   'mobile_vit_tiny_novit_test5','mobile_vit_tiny_novit_test6','mobile_vit_tiny_novit_test0','mobile_vit_tiny_dilatedblock_5','mobile_vit_tiny_dilatedblock_7',
                                   'mobile_vit_tiny_dilatedblock_9','mobile_vit_tiny_dilatedblock_5_1','mobile_vit_tiny_dilatedblock_5_2','mobile_vit_tiny_dilatedblock_5_3','mobile_vit_tiny_dilatedblock_5_4','mobile_vit_tiny_dilatedblock_5_5',
                                   'mobile_vit_tiny_dilatedblock_5_noweight','mcunet','SwiftFormer_XXS','edgenext_xxx_small','mobilecontainer','repvit_m0_6','repvit_m0_6_infiuni','RepInfiniteUniVit_initial','repvit_m0_6_uni_ours',
                                   'mobile_half_1_1_1','mobile_half_1_1_2','mobile_half_1_1_3','mobile_half_1_1_4','mobile_half_1_2_1','mobile_half_1_2_2','mobile_half_1_2_3','mobile_half_1_2_4','mobile_half_1_3_1','mobile_half_1_1_5',
                                   'mobile_half_2_1_1','mobile_half_2_1_1_1','mobile_half_2_1_2','mobile_half_2_1_3','mobile_half_2_1_4','mobile_half_2_1_5','mobile_half_2_1_6','mobile_half_3_1_1',
                                   'mobile_half_4_1_1','mobile_half_4_1_2','mobile_half_4_1_3','mobile_half_4_1_4','mobile_half_4_1_5','mobile_half_4_1_6','mobile_half_4_1_7','mobile_half_5_1_1','mobile_half_percent',
                                   'mobile_half_1_1_7','mobile_half_1_1_8','mobile_half_1_1_9','mobile_half_1_1_1_1','mobile_half_1_1_10','mobile_half_1_1_11','mobile_half_1_1_12','mobile_half_4_1_2_1','mobile_half_4_1_2_2','mobile_half_4_1_3_1','mobile_half_4_1_3_3','mobile_half_4_1_3_2',
                                   'mobile_half_1_1_8_1','mobile_half_1_1_8_2','mobile_half_6_1_1_1','mobile_half_6_1_1_2','mobile_half_6_1_2_1','mobile_half_1_1_12_1','mobile_half_1_1_8_3','mobile_half_6_1_1_2_1','mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1',
                                   'mobile_half_1_2_2_1','mobile_half_1_2_1_1','mobile_half_7_1_1','mobile_half_7_1_2','mobile_half_7_1_1_1','mobile_half_7_1_2_1','mobile_half_1_1_1_2','mobile_half_5_1_1_1','mobile_half_5_1_1_2','mobile_half_1_1_8_4','mobile_half_1_1_8_5','mobile_half_1_1_8_6','mobile_half_1_1_8_7',
                                   'mobilemetanet','mobilemetanet_1','mobilemetanet_2','mobile_half_3_1_2','RepTinynet','RepTinynet1','RepTinynet2','RepTinynet3','RepTinynet4','RepTinynet5','RepTinynet6','RepTinynet7','RepTinynet8','RepTinynet9','RepTinynet10','RepTinynet11','RepTinynet12','RepTinynet13','mcunetlike','RepTinynet14','RepTinynet15','RepTinynet16','RepTinynet17',
                                   'mobile_half_5_1_1_3','mobile_half_class','mobile_half_class_2','mobile_half_class_3','RepTinynet18','RepTinynet19','RepTinynet20','RepTinynet21','RepTinynet22','RepTinynet23','mobile_half_5_1_1_4','RepTinynet24','RepTinynet25','repvggnet','McuNetv1','finalNet'])
    parser.add_argument('--dataset', type=str, default='miniImage', choices=['cifar100','miniImage'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('-cuda', '--cuda', type=str, default='0', help='the cuda number')
    parser.add_argument('--arch_name', type=str, default='mobile_vit_tiny_likevitpp',
                        help='log name')
    parser.add_argument('--OUTPUT', type=str, default='./log',
                        help='log output path')

    opt = parser.parse_args()
    print('111'+opt.model)
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2','MobileNetV3' 'ShuffleV1', 'ShuffleV2','mobile_vit_tiny','mobile_vit_xx_small_init','mobile_vit_x_small_init',
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
                                   'mobile_half_5_1_1_3','mobile_half_class','mobile_half_class_2','mobile_half_class_3','RepTinynet18','RepTinynet19','RepTinynet20','RepTinynet21','RepTinynet22','RepTinynet23','mobile_half_5_1_1_4','RepTinynet24','RepTinynet25','repvggnet','McuNetv1','finalNet']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('wsl'):
        opt.model_path = './path/teacher_model'
        opt.tb_path = './path/teacher_tensorboards'
    else:
        opt.model_path = './save/teacher_model'
        opt.tb_path = './save/teacher_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


class NHWCWrapper(nn.Module):
    def __init__(self, nchw_model):
        super().__init__()
        self.model = nchw_model

    def forward(self, x_nhwc):
        # x_nhwc is [B, H, W, C]
        x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
        y = self.model(x_nchw)

        if y.dim() == 4:
            # a feature‐map; do NCHW -> NHWC
            return y.permute(0, 2, 3, 1).contiguous()
        else:
            # a vector or any other rank; just pass it through
            return y

def main():
    np.random.seed(970203)
    torch.manual_seed(970203)
    torch.cuda.manual_seed_all(970203) #所有GPU
    torch.cuda.manual_seed(970203)     # 当前GPU
    best_acc = 0
    opt = parse_option()
    device = torch.device("cuda:"+opt.cuda if torch.cuda.is_available() else "cpu")
    
    



    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'miniImage':
        train_loader, val_loader = get_miniImagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)


    model = model_dict[opt.model](num_classes=n_cls)

    if opt.weights != '':
        print('finetune')
        if os.path.exists(opt.weights):
            assert os.path.exists(opt.weights), "file {} does not exist.".format(opt.weights)
            # net.load_state_dict(torch.load(args.weights, map_location='cpu'))

            weights_dict = torch.load(opt.weights,map_location=device)

            # cycleer = itertools.cycle(list(weights_dict['model'].keys()))
            # cycleer_asis = itertools.cycle(list(weights_dict['model'].keys())) 
            # load_weights_dict = {k: weights_dict[next(cycleer)]  if 'repadd' not in k  and weights_dict[next(cycleer_asis)].numel() == model.state_dict()[k].numel() else v for k, v, in model.state_dict().items()}                                
            model.load_state_dict(weights_dict['model'], strict=True)
            print('model successful load')
           

    # optimizer
  

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

    # tensorboard
    

    
    acc_arr = []
    acc5_arr = []
    loss_arr = []
    throughout1 = []
    test_acc, test_acc_top5,  = validate(val_loader, model, criterion, opt,device)
    acc_arr.append(float(test_acc))
    acc5_arr.append(float(test_acc_top5))

    print(acc_arr)
    print(acc5_arr)
    print(loss_arr)

    print('====')

    model.eval()

    # transforming models from training to inference version 
    acc_arr = []
    acc5_arr = []
    loss_arr = []
    throughout1 = []
    total_params = sum(p.numel() for p in model.parameters())
    print('total_params_before_merge:'+str(total_params))
    model.train()
    model.merge_rep_armor_mbv2()
    model.eval()
    total_params2 = sum(p.numel() for p in model.parameters())
    print('total_params_after_merge:'+str(total_params2))
    test_acc, test_acc_top5,  = validate(val_loader, model, criterion, opt,device)
    acc_arr.append(float(test_acc))
    acc5_arr.append(float(test_acc_top5))

    print(acc_arr)
    print(acc5_arr)

    model.eval()
    model.to(memory_format=torch.channels_last)
    for k, v in model.named_parameters():

        if v.dim() == 4:  
            v.data = v.data.to(memory_format=torch.channels_last)


    dummy_input = torch.randn(1,224, 224,3).to(device)  # input size
    dummy_input = dummy_input.to(memory_format=torch.channels_last)
    wrapper = NHWCWrapper(model)
    wrapper.eval()

    torch.onnx.export(
    wrapper,
    # model,
    dummy_input,
    "final_r2.onnx",
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],

    keep_initializers_as_inputs=True,
    export_params=True,
    # dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},  # adaptive batch, not support for MCUs
    # dynamic_axes={ "input": {0: "batch"} }
)

if __name__ == '__main__':
    main()
