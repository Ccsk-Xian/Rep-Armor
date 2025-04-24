# # # import torch
# # # from models.mobilevit import mobile_vit_small,mobile_vit_xx_small,mobile_vit_x_small,mobile_vit_tiny_init
# # # from dataset.cifar100 import get_cifar100_dataloaders
# # # from helper.util import  accuracy, AverageMeter
# # # import time

# # # def validate(val_loader, model, criterion):
# # #     """validation"""
# # #     batch_time = AverageMeter()
# # #     losses = AverageMeter()
# # #     top1 = AverageMeter()
# # #     top5 = AverageMeter()

# # #     # switch to evaluate mode
# # #     model.eval()

# # #     with torch.no_grad():
# # #         end = time.time()
# # #         for idx, (input, target) in enumerate(val_loader):

# # #             input = input.float()
# # #             if torch.cuda.is_available():
# # #                 input = input.cuda()
# # #                 target = target.cuda()

# # #             # compute output
# # #             output = model(input)
# # #             loss = criterion(output, target)

# # #             # measure accuracy and record loss
# # #             acc1, acc5 = accuracy(output, target, topk=(1, 5))
# # #             losses.update(loss.item(), input.size(0))
# # #             top1.update(acc1[0], input.size(0))
# # #             top5.update(acc5[0], input.size(0))

# # #             # measure elapsed time
# # #             batch_time.update(time.time() - end)
# # #             end = time.time()

# # #             if idx % 40 == 0:
# # #                 print('Test: [{0}/{1}]\t'
# # #                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
# # #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# # #                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
# # #                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
# # #                        idx, len(val_loader), batch_time=batch_time, loss=losses,
# # #                        top1=top1, top5=top5))

# # #         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
# # #               .format(top1=top1, top5=top5))

# # #     return top1.avg, top5.avg, losses.avg
        
# # # train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=224,
# # #                                                             num_workers=8,
# # #                                                             is_instance=True)
# # # model = mobile_vit_xx_small(num_classes=100)
# # # model.load_state_dict(torch.load('/root/distill/path/teacher_model/mobile_vit_xx_small_cifar100_lr_0.05_decay_0.0005_trial_0/mobile_vit_xx_small_best.pth')['model'])
# # # criterion_cls = torch.nn.CrossEntropyLoss()
# # # model = model.cuda()
# # # teacher_acc, _, _ = validate(val_loader, model, criterion_cls)
# # # print('teacher accuracy: ', teacher_acc)

# # # import torch
# # # a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]]).reshape(1,1,3,3)
# # # conv1 = torch.nn.Conv2d(1,2,1,1,0,dilation=8,bias=False)
# # # torch.nn.init.constant_(conv1.weight.data,1.0)
# # # out = conv1(a)
# # # print(conv1.weight.data)
# # # print(out)


# # # import torch
# # # # import torch.nn

# # # a = torch.linspace(0,1,20)
# # # act = torch.nn.LogSigmoid()
# # # print(-act(a))

# # # def find_common_factors_and_select(a, b,c):

# # #     min_num = min(a, b)
# # #     common_factors = []
# # #     for i in range(1, min_num + 1):
# # #         if a % i == 0 and b % i == 0:
# # #             common_factors.append(i)


# # #     common_factors.sort()
# # #     if len(common_factors)<=c:
# # #         return common_factors

# # #     step = len(common_factors) / (c-1)



# # #     selected_factors = []
# # #     selected_factors.append(common_factors[0])
# # #     selected_factors.append(common_factors[-1])
# # #     for i in range(c-2):
# # #         selected_factors.append(common_factors[int((i+1)*step)])

# # #     return selected_factors

# # # c = find_common_factors_and_select(25,100,4)
# # # print(c)
# # # import torch

# # # a = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.])
# # # b = a.view(2,3,1,2)
# # # print(b)
# # # c = torch.softmax(b,dim=-1)
# # # print(c)
# # # d = torch.tensor([1.,2.])
# # # e = torch.softmax(d,dim=0)
# # # print(e)

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import time
# # import numpy as np

# # def conv_pure(in_channels, out_channels, kernel_size, stride, padding, groups=1):
# #     result = nn.Sequential()
# #     result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
# #                                         kernel_size=kernel_size, stride=stride, padding=padding,
# #                                         groups=groups, bias=False))
# #     result.add_module('conv_1', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
# #                                         kernel_size=1, stride=1, padding=0,
# #                                         groups=groups, bias=False))
   
# #     return result


# # # class RepVGGBlock(nn.Module):
# # #     def __init__(self, in_channels, out_channels, kernel_size=1,
# # #                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
# # #         super(RepVGGBlock, self).__init__()
# # #         self.deploy = deploy
# # #         self.groups = groups
# # #         self.in_channels = in_channels
# # #         self.nonlinearity = nn.ReLU()

# # #         if deploy:
# # #             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
# # #                                          kernel_size=kernel_size, stride=stride,
# # #                                          padding=padding, dilation=dilation, groups=groups,
# # #                                          bias=False, padding_mode=padding_mode)

# # #         else:
# # #             self.rbr_dense = conv_pure(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
# # #                                      stride=stride, padding=padding, groups=groups)
# # #             self.rbr_1x1 = conv_pure(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
# # #                                    stride=stride, padding=0, groups=groups)

# # #     def forward(self, inputs):
# # #         if hasattr(self, 'rbr_reparam'):
# # #             return self.nonlinearity(self.rbr_reparam(inputs))

        

# # #         return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs))
    
# # #     def get_equivalent_kernel_bias(self):
# # #         # torch.einsum('bij,bjk->bik', As, Bs)
# # #         # torch.einsum('bchw,')
# # #         # kernel3x3 = torch.matmul(self.rbr_dense.conv.weight.data.view(64,64).transpose(0,1),self.rbr_dense.conv_1.weight.data.view(64,64).transpose(0,1)).view(64,64,1,1).transpose(0,1)
# # #         # kernel1x1 = torch.matmul(self.rbr_1x1.conv.weight.data.view(64,64).transpose(0,1),self.rbr_1x1.conv_1.weight.data.view(64,64).transpose(0,1)).view(64,64,1,1).transpose(0,1)
# # #         kernel3x3 = F.conv2d(self.rbr_dense.conv.weight.data.transpose(0,1),self.rbr_dense.conv_1.weight.data).transpose(0,1)
# # #         # kernel3x3 = torch.einsum('ijkl,jskl->iskl',self.rbr_dense.conv_1.weight.data,self.rbr_dense.conv.weight.data)
# # #         # kernel3x3 = torch.matmul(self.rbr_dense.conv_1.weight.data.view(64,64),self.rbr_dense.conv.weight.data.view(64,64)).view(64,64,1,1)
# # #         kernel1x1 = F.conv2d(self.rbr_1x1.conv.weight.data.transpose(0,1),self.rbr_1x1.conv_1.weight.data).transpose(0,1)
# # #         # kernel1x1 = torch.einsum('ijkl,jskl->iskl',self.rbr_1x1.conv_1.weight.data,self.rbr_1x1.conv.weight.data)
# # #         # kernel1x1 = torch.matmul(self.rbr_1x1.conv_1.weight.data.view(64,64),self.rbr_1x1.conv.weight.data.view(64,64)).view(64,64,1,1)
# # #         # kernel3x3 = F.conv2d(self.rbr_dense.conv.weight.data,self.rbr_dense.conv_1.weight.data,stride=1,padding=0,groups=self.groups)
        
# # #         # kernel1x1 = F.conv2d(self.rbr_1x1.conv.weight.data,self.rbr_1x1.conv_1.weight.data,stride=1,padding=0,groups=self.groups)
# # #         # kernel3x3 = self.rbr_dense.conv.weight+self.rbr_dense.conv_1.weight
# # #         # kernel1x1 = self.rbr_1x1.conv.weight+self.rbr_1x1.conv_1.weight


# # #         # return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
# # #         return kernel3x3 + kernel1x1
    
# # #     def switch_to_deploy(self):
# # #         if hasattr(self, 'rbr_reparam'):
# # #             return
# # #         kernel = self.get_equivalent_kernel_bias()
# # #         # print(kernel.shape)
# # #         self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
# # #                                      out_channels=self.rbr_dense.conv.out_channels,
# # #                                      kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
# # #                                      padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
# # #                                      groups=self.rbr_dense.conv.groups, bias=False)
# # #         self.rbr_reparam.weight.data = kernel
# # #         for para in self.parameters():
# # #             para.detach_()
# # #         self.__delattr__('rbr_dense')
# # #         self.__delattr__('rbr_1x1')
# # #         if hasattr(self, 'rbr_identity'):
# # #             self.__delattr__('rbr_identity')
# # #         if hasattr(self, 'id_tensor'):
# # #             self.__delattr__('id_tensor')
# # #         self.deploy = True



# # # def main():
# # #     f1 = torch.randn(1, 64, 64, 64)
# # #     block = RepVGGBlock(in_channels=64, out_channels=64)
# # #     block.eval()
# # #     with torch.no_grad():
# # #         output1 = block(f1)
# # #         start_time = time.time()
# # #         for _ in range(100):
# # #             block(f1)
# # #         print(f"consume time: {time.time() - start_time}")

# # #         # re-parameterization
# # #         block.switch_to_deploy()
# # #         output2 = block(f1)
# # #         start_time = time.time()
# # #         for _ in range(100):
# # #             block(f1)
# # #         print(f"consume time: {time.time() - start_time}")

# # #         np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
# # #         print("convert module has been tested, and the result looks good!")


# # # if __name__ == '__main__':
# # #     main()

# # import torch
# # import torch.nn.functional as F
# # # a = torch.tensor([[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0],[0,1,1,0,1,2,2,1,0],[1,1,1,0,2,3,3,1,0]]).reshape(2,2,3,3)

# # # b = torch.tensor([[2,4],[1,3]]).reshape(2,2,1,1)
# # input1  = torch.tensor([[3,1,2,4,2,6,8,4,9]]).reshape(1,1,3,3)
# # # input1  = torch.tensor([[3,1,2,4,2,6,8,4,9]]).reshape(1,1,3,3)
# # # kernel_a = torch.tensor([[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0],[0,1,1,0,1,2,2,1,0],[1,1,1,0,2,3,3,1,0]]).reshape(2,2,3,3)

# # kernel_a = torch.tensor([[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0]]).reshape(2,1,3,3)
# # kernel_b = torch.tensor([[1,0],[3,4],[1,4],[3,2]]).reshape(4,2,1,1)
# # kernel_b_c = torch.tensor([[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0],[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0],[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0],[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,3,1,1,0],[1,1,1,0,2,3,3,1,0],[0,1,0,5,2,3,1,1,0],[1,1,1,0,2,3,3,1,0],[0,1,0,2,1,8,1,1,0],[1,1,1,0,2,3,3,1,0],[5,1,1,2,1,3,1,1,0],[1,1,1,4,2,3,3,1,2],[0,1,0,3,1,3,1,1,0]]).reshape(4,4,3,3)
# # kernel_c = torch.tensor([[1,0],[3,4],[1,4],[3,2]]).reshape(2,4,1,1)
# # # kernel_b = torch.tensor([[1,0,1,0,1,1,2,1,0],[0,1,0,1,0,2,1,1,0],[0,1,1,0,0,1,2,1,0],[1,1,1,0,1,3,2,1,0]]).reshape(2,2,3,3)
# # a = torch.tensor([[1,1,1,0,2,3,3,1,0],[0,1,1,0,1,2,2,1,0]]).reshape(1,2,3,3)
# # # a = torch.tensor([[0,1,0,2,1,3,1,1,0],[1,1,1,0,2,3,3,1,0]]).reshape(1,2,3,3)
# # b = torch.tensor([[1,0,1,0,1,1,2,1,0],[0,1,0,1,0,2,1,1,0]]).reshape(1,2,3,3)
# # # b = torch.tensor([[1,0,1,0,1,1,2,1,0],[0,1,0,1,0,2,1,1,0]]).reshape(1,2,3,3)
# # # b1 = torch.flip(b,dims=(3,))
# # # print(b)
# # # print(b1)
# # # a = torch.tensor([[1,1,1,0,2,3,3,1,0],[0,1,1,0,1,2,2,1,0],[]]).reshape(1,2,3,3)
# # # kernel_b = torch.tensor([[2,4],[1,3]]).reshape(2,2,1,1)
# # # b = b.expand(-1,-1,3,3)
# # d = torch.einsum('ijkl,jskl->iskl',kernel_b,kernel_a)
# # d = torch.einsum('ijkl,jskl->iskl',kernel_b_c,d)
# # d = torch.einsum('ijkl,jskl->iskl',kernel_c,d)
# # # d = F.conv2d(kernel_a.transpose(0,1),kernel_b,padding=0).transpose(0,1)
# # # print(d.shape)
# # # input1 = F.conv2d(input1,kernel_a,padding=1,stride=2)
# # # input1 = F.conv2d(input1,kernel_a,padding=1,stride=1)
# # # print(input1.shape)
# # # input1 = F.conv2d(input1,kernel_b,padding=0)
# # # print(input1.shape)
# # # # kernel_b_c
# # # input1 = F.conv2d(input1,kernel_b_c,padding=1)
# # # print(input1.shape)
# # # result = F.conv2d(input1,kernel_c,padding=0)
# # # print(result.shape)
# # print(d.shape)
# # result = F.conv2d(input1,d,padding=1,stride=1)
# # print(result)
# # # kernel = torch.einsum('ijdw,jxws->ixds',kernel_b,kernel_a.transpose(2,3))
# # # 'ijdw,ixws->ixds' 'ijdw,ijws->ijds'
# # # kernel = torch.einsum('ijdw,jqws->iqds',a1,b)
# # # ker = torch.einsum('ijdw,jqws->iqds',a,b)
# # # d = F.conv2d(input1,d,padding=1)
# # # print(kernel)
# # # print(ker)
# # # kernel = torch.einsum('ijdw,jskl->iskl',b,a)
# # # print(kernel)

# # # import torch
# # # import timm
# # # timm.models.efficientvit_mit
# # # drop = 0.1
# # # x = torch.rand(10, 1, 1, 1,
# # #                                               ).ge_(drop).div(1 - drop)
# # # print(x)
# # import torch

# # weights_dict = torch.load('/root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/ckpt_epoch_240.pth')
# # print(weights_dict.keys())
# # if  'cc' in weights_dict.keys():
# #     print(1) 
# # print(weights_dict['model'].keys())

# # # print(weights_dict['model']['blocks.0.0.dw.rbr_dense_2.bn.weight'])
# # # print(weights_dict['model']['blocks.0.0.dw.rbr_dense_2.conv1.needle.3.weight'])
# # # print(weights_dict['model']['conv1.origin_1x1.needle.3.weight'])
# # # print(weights_dict['model']['conv1.origin_bn.weight'])
# # # print(weights_dict['model']['conv1.dil_bn_k5_1.weight'])
# # # print(weights_dict['model']['conv1_follow.1.weight'])
# # # # print(weights_dict['model']['blocks.6.0.dw.rbr_dense_2.bn.weight'])
# # # # print(weights_dict['model']['blocks.6.0.dw.rbr_dense_2.conv1.needle.3.weight'])
# # # print(weights_dict['model']['blocks.1.0.dw.rbr_dense_2.conv1.needle.3.weight'])
# # # print(weights_dict['model']['blocks.1.0.dw.rbr_dense_2.bn.weight'])
# # print(weights_dict['best_acc'])

# # print(weights_dict['optimizer'])
# # print(weights_dict['opt'])


# # odict_keys(['conv1.lk_origin.weight', 'conv1.origin_1x1.needle.0.weight', 'conv1.origin_1x1.needle.1.weight', 'conv1.origin_1x1.needle.2.weight', 'conv1.origin_1x1.needle.3.weight', 'conv1.origin_1x1.needle.3.bias', 'conv1.origin_1x1.needle.3.running_mean', 'conv1.origin_1x1.needle.3.running_var', 'conv1.origin_1x1.needle.3.num_batches_tracked', 'conv1.origin_bn.weight', 'conv1.origin_bn.bias', 'conv1.origin_bn.running_mean', 'conv1.origin_bn.running_var', 'conv1.origin_bn.num_batches_tracked', 'conv1.dil_conv_k5_1.weight', 'conv1.dil_bn_k5_1.weight', 'conv1.dil_bn_k5_1.bias', 'conv1.dil_bn_k5_1.running_mean', 'conv1.dil_bn_k5_1.running_var', 'conv1.dil_bn_k5_1.num_batches_tracked', 'conv1.conv_one5_1.needle.0.weight', 'conv1.conv_one5_1.needle.1.weight', 'conv1.conv_one5_1.needle.2.weight', 'conv1.conv_one5_1.needle.3.weight', 'conv1.conv_one5_1.needle.3.bias', 'conv1.conv_one5_1.needle.3.running_mean', 'conv1.conv_one5_1.needle.3.running_var', 'conv1.conv_one5_1.needle.3.num_batches_tracked', 'conv1.dil_conv_k3_2.weight', 'conv1.dil_bn_k3_2.weight', 'conv1.dil_bn_k3_2.bias', 'conv1.dil_bn_k3_2.running_mean', 'conv1.dil_bn_k3_2.running_var', 'conv1.dil_bn_k3_2.num_batches_tracked', 'conv1.conv_one3_2.needle.0.weight', 'conv1.conv_one3_2.needle.1.weight', 'conv1.conv_one3_2.needle.2.weight', 'conv1.conv_one3_2.needle.3.weight', 'conv1.conv_one3_2.needle.3.bias', 'conv1.conv_one3_2.needle.3.running_mean', 'conv1.conv_one3_2.needle.3.running_var', 'conv1.conv_one3_2.needle.3.num_batches_tracked', 'conv1.dil_conv_k3_3.weight', 'conv1.dil_bn_k3_3.weight', 'conv1.dil_bn_k3_3.bias', 'conv1.dil_bn_k3_3.running_mean', 'conv1.dil_bn_k3_3.running_var', 'conv1.dil_bn_k3_3.num_batches_tracked', 'conv1.conv_one3_3.needle.0.weight', 'conv1.conv_one3_3.needle.1.weight', 'conv1.conv_one3_3.needle.2.weight', 'conv1.conv_one3_3.needle.3.weight', 'conv1.conv_one3_3.needle.3.bias', 'conv1.conv_one3_3.needle.3.running_mean', 'conv1.conv_one3_3.needle.3.running_var', 'conv1.conv_one3_3.needle.3.num_batches_tracked', 'conv1.dil_conv_k3_1.weight', 'conv1.dil_bn_k3_1.weight', 'conv1.dil_bn_k3_1.bias', 'conv1.dil_bn_k3_1.running_mean', 'conv1.dil_bn_k3_1.running_var', 'conv1.dil_bn_k3_1.num_batches_tracked', 'conv1.conv_one3_1.needle.0.weight', 'conv1.conv_one3_1.needle.1.weight', 'conv1.conv_one3_1.needle.2.weight', 'conv1.conv_one3_1.needle.3.weight', 'conv1.conv_one3_1.needle.3.bias', 'conv1.conv_one3_1.needle.3.running_mean', 'conv1.conv_one3_1.needle.3.running_var', 'conv1.conv_one3_1.needle.3.num_batches_tracked', 'conv1_follow.0.weight', 'conv1_follow.1.weight', 'conv1_follow.1.bias', 'conv1_follow.1.running_mean', 'conv1_follow.1.running_var', 'conv1_follow.1.num_batches_tracked', 'blocks.0.0.dw.rbr_dense.conv.weight', 'blocks.0.0.dw.rbr_dense.bn.weight', 'blocks.0.0.dw.rbr_dense.bn.bias', 'blocks.0.0.dw.rbr_dense.bn.running_mean', 'blocks.0.0.dw.rbr_dense.bn.running_var', 'blocks.0.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.0.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.0.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.0.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.0.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.0.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.0.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.0.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.0.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.0.0.dw.rbr_dense_2.conv.weight', 'blocks.0.0.dw.rbr_dense_2.bn.weight', 'blocks.0.0.dw.rbr_dense_2.bn.bias', 'blocks.0.0.dw.rbr_dense_2.bn.running_mean', 'blocks.0.0.dw.rbr_dense_2.bn.running_var', 'blocks.0.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.0.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.0.0.pw.0.weight', 'blocks.0.0.pw.1.weight', 'blocks.0.0.pw.1.bias', 'blocks.0.0.pw.1.running_mean', 'blocks.0.0.pw.1.running_var', 'blocks.0.0.pw.1.num_batches_tracked', 'blocks.0.0.pw.2.needle.0.weight', 'blocks.0.0.pw.2.needle.1.weight', 'blocks.0.0.pw.2.needle.2.weight', 'blocks.0.0.pw.2.needle.3.weight', 'blocks.0.0.pw.2.needle.3.bias', 'blocks.0.0.pw.2.needle.3.running_mean', 'blocks.0.0.pw.2.needle.3.running_var', 'blocks.0.0.pw.2.needle.3.num_batches_tracked', 'blocks.0.0.conv.0.weight', 'blocks.0.0.conv.1.weight', 'blocks.0.0.conv.1.bias', 'blocks.0.0.conv.1.running_mean', 'blocks.0.0.conv.1.running_var', 'blocks.0.0.conv.1.num_batches_tracked', 'blocks.1.0.dw.rbr_dense.conv.weight', 'blocks.1.0.dw.rbr_dense.bn.weight', 'blocks.1.0.dw.rbr_dense.bn.bias', 'blocks.1.0.dw.rbr_dense.bn.running_mean', 'blocks.1.0.dw.rbr_dense.bn.running_var', 'blocks.1.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.1.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.1.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.1.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.1.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.1.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.1.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.1.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.1.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.1.0.dw.rbr_dense_2.conv.weight', 'blocks.1.0.dw.rbr_dense_2.bn.weight', 'blocks.1.0.dw.rbr_dense_2.bn.bias', 'blocks.1.0.dw.rbr_dense_2.bn.running_mean', 'blocks.1.0.dw.rbr_dense_2.bn.running_var', 'blocks.1.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.1.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.1.0.pw.0.weight', 'blocks.1.0.pw.1.weight', 'blocks.1.0.pw.1.bias', 'blocks.1.0.pw.1.running_mean', 'blocks.1.0.pw.1.running_var', 'blocks.1.0.pw.1.num_batches_tracked', 'blocks.1.0.pw.2.needle.0.weight', 'blocks.1.0.pw.2.needle.1.weight', 'blocks.1.0.pw.2.needle.2.weight', 'blocks.1.0.pw.2.needle.3.weight', 'blocks.1.0.pw.2.needle.3.bias', 'blocks.1.0.pw.2.needle.3.running_mean', 'blocks.1.0.pw.2.needle.3.running_var', 'blocks.1.0.pw.2.needle.3.num_batches_tracked', 'blocks.1.0.conv.0.weight', 'blocks.1.0.conv.1.weight', 'blocks.1.0.conv.1.bias', 'blocks.1.0.conv.1.running_mean', 'blocks.1.0.conv.1.running_var', 'blocks.1.0.conv.1.num_batches_tracked', 'blocks.1.1.dw.rbr_dense.conv.weight', 'blocks.1.1.dw.rbr_dense.bn.weight', 'blocks.1.1.dw.rbr_dense.bn.bias', 'blocks.1.1.dw.rbr_dense.bn.running_mean', 'blocks.1.1.dw.rbr_dense.bn.running_var', 'blocks.1.1.dw.rbr_dense.bn.num_batches_tracked', 'blocks.1.1.dw.rbr_dense.conv1.needle.0.weight', 'blocks.1.1.dw.rbr_dense.conv1.needle.1.weight', 'blocks.1.1.dw.rbr_dense.conv1.needle.2.weight', 'blocks.1.1.dw.rbr_dense.conv1.needle.3.weight', 'blocks.1.1.dw.rbr_dense.conv1.needle.3.bias', 'blocks.1.1.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.1.1.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.1.1.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.1.1.dw.rbr_dense_2.conv.weight', 'blocks.1.1.dw.rbr_dense_2.bn.weight', 'blocks.1.1.dw.rbr_dense_2.bn.bias', 'blocks.1.1.dw.rbr_dense_2.bn.running_mean', 'blocks.1.1.dw.rbr_dense_2.bn.running_var', 'blocks.1.1.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.1.1.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.1.1.pw.0.weight', 'blocks.1.1.pw.1.weight', 'blocks.1.1.pw.1.bias', 'blocks.1.1.pw.1.running_mean', 'blocks.1.1.pw.1.running_var', 'blocks.1.1.pw.1.num_batches_tracked', 'blocks.1.1.pw.2.needle.0.weight', 'blocks.1.1.pw.2.needle.1.weight', 'blocks.1.1.pw.2.needle.2.weight', 'blocks.1.1.pw.2.needle.3.weight', 'blocks.1.1.pw.2.needle.3.bias', 'blocks.1.1.pw.2.needle.3.running_mean', 'blocks.1.1.pw.2.needle.3.running_var', 'blocks.1.1.pw.2.needle.3.num_batches_tracked', 'blocks.1.1.conv.0.weight', 'blocks.1.1.conv.1.weight', 'blocks.1.1.conv.1.bias', 'blocks.1.1.conv.1.running_mean', 'blocks.1.1.conv.1.running_var', 'blocks.1.1.conv.1.num_batches_tracked', 'blocks.2.0.dw.rbr_dense.conv.weight', 'blocks.2.0.dw.rbr_dense.bn.weight', 'blocks.2.0.dw.rbr_dense.bn.bias', 'blocks.2.0.dw.rbr_dense.bn.running_mean', 'blocks.2.0.dw.rbr_dense.bn.running_var', 'blocks.2.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.2.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.2.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.2.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.2.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.2.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.2.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.2.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.2.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.2.0.dw.rbr_dense_2.conv.weight', 'blocks.2.0.dw.rbr_dense_2.bn.weight', 'blocks.2.0.dw.rbr_dense_2.bn.bias', 'blocks.2.0.dw.rbr_dense_2.bn.running_mean', 'blocks.2.0.dw.rbr_dense_2.bn.running_var', 'blocks.2.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 
# # 'blocks.2.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.2.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.2.0.pw.0.weight', 'blocks.2.0.pw.1.weight', 'blocks.2.0.pw.1.bias', 'blocks.2.0.pw.1.running_mean', 'blocks.2.0.pw.1.running_var', 'blocks.2.0.pw.1.num_batches_tracked', 'blocks.2.0.pw.2.needle.0.weight', 'blocks.2.0.pw.2.needle.1.weight', 'blocks.2.0.pw.2.needle.2.weight', 'blocks.2.0.pw.2.needle.3.weight', 'blocks.2.0.pw.2.needle.3.bias', 'blocks.2.0.pw.2.needle.3.running_mean', 'blocks.2.0.pw.2.needle.3.running_var', 'blocks.2.0.pw.2.needle.3.num_batches_tracked', 'blocks.2.0.conv.0.weight', 'blocks.2.0.conv.1.weight', 'blocks.2.0.conv.1.bias', 'blocks.2.0.conv.1.running_mean', 'blocks.2.0.conv.1.running_var', 'blocks.2.0.conv.1.num_batches_tracked', 'blocks.2.1.dw.rbr_dense.conv.weight', 'blocks.2.1.dw.rbr_dense.bn.weight', 'blocks.2.1.dw.rbr_dense.bn.bias', 'blocks.2.1.dw.rbr_dense.bn.running_mean', 'blocks.2.1.dw.rbr_dense.bn.running_var', 'blocks.2.1.dw.rbr_dense.bn.num_batches_tracked', 'blocks.2.1.dw.rbr_dense.conv1.needle.0.weight', 'blocks.2.1.dw.rbr_dense.conv1.needle.1.weight', 'blocks.2.1.dw.rbr_dense.conv1.needle.2.weight', 'blocks.2.1.dw.rbr_dense.conv1.needle.3.weight', 'blocks.2.1.dw.rbr_dense.conv1.needle.3.bias', 'blocks.2.1.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.2.1.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.2.1.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.2.1.dw.rbr_dense_2.conv.weight', 'blocks.2.1.dw.rbr_dense_2.bn.weight', 'blocks.2.1.dw.rbr_dense_2.bn.bias', 'blocks.2.1.dw.rbr_dense_2.bn.running_mean', 'blocks.2.1.dw.rbr_dense_2.bn.running_var', 'blocks.2.1.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.2.1.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.2.1.pw.0.weight', 'blocks.2.1.pw.1.weight', 'blocks.2.1.pw.1.bias', 'blocks.2.1.pw.1.running_mean', 'blocks.2.1.pw.1.running_var', 'blocks.2.1.pw.1.num_batches_tracked', 'blocks.2.1.pw.2.needle.0.weight', 'blocks.2.1.pw.2.needle.1.weight', 'blocks.2.1.pw.2.needle.2.weight', 'blocks.2.1.pw.2.needle.3.weight', 'blocks.2.1.pw.2.needle.3.bias', 'blocks.2.1.pw.2.needle.3.running_mean', 'blocks.2.1.pw.2.needle.3.running_var', 'blocks.2.1.pw.2.needle.3.num_batches_tracked', 'blocks.2.1.conv.0.weight', 'blocks.2.1.conv.1.weight', 'blocks.2.1.conv.1.bias', 'blocks.2.1.conv.1.running_mean', 'blocks.2.1.conv.1.running_var', 'blocks.2.1.conv.1.num_batches_tracked', 'blocks.2.2.dw.rbr_dense.conv.weight', 'blocks.2.2.dw.rbr_dense.bn.weight', 'blocks.2.2.dw.rbr_dense.bn.bias', 'blocks.2.2.dw.rbr_dense.bn.running_mean', 'blocks.2.2.dw.rbr_dense.bn.running_var', 'blocks.2.2.dw.rbr_dense.bn.num_batches_tracked', 'blocks.2.2.dw.rbr_dense.conv1.needle.0.weight', 'blocks.2.2.dw.rbr_dense.conv1.needle.1.weight', 'blocks.2.2.dw.rbr_dense.conv1.needle.2.weight', 'blocks.2.2.dw.rbr_dense.conv1.needle.3.weight', 'blocks.2.2.dw.rbr_dense.conv1.needle.3.bias', 'blocks.2.2.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.2.2.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.2.2.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.2.2.dw.rbr_dense_2.conv.weight', 'blocks.2.2.dw.rbr_dense_2.bn.weight', 'blocks.2.2.dw.rbr_dense_2.bn.bias', 'blocks.2.2.dw.rbr_dense_2.bn.running_mean', 'blocks.2.2.dw.rbr_dense_2.bn.running_var', 'blocks.2.2.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.2.2.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.2.2.pw.0.weight', 'blocks.2.2.pw.1.weight', 'blocks.2.2.pw.1.bias', 'blocks.2.2.pw.1.running_mean', 'blocks.2.2.pw.1.running_var', 'blocks.2.2.pw.1.num_batches_tracked', 'blocks.2.2.pw.2.needle.0.weight', 'blocks.2.2.pw.2.needle.1.weight', 'blocks.2.2.pw.2.needle.2.weight', 'blocks.2.2.pw.2.needle.3.weight', 'blocks.2.2.pw.2.needle.3.bias', 'blocks.2.2.pw.2.needle.3.running_mean', 'blocks.2.2.pw.2.needle.3.running_var', 'blocks.2.2.pw.2.needle.3.num_batches_tracked', 'blocks.2.2.conv.0.weight', 'blocks.2.2.conv.1.weight', 'blocks.2.2.conv.1.bias', 'blocks.2.2.conv.1.running_mean', 'blocks.2.2.conv.1.running_var', 'blocks.2.2.conv.1.num_batches_tracked', 'blocks.3.0.dw.rbr_dense.conv.weight', 'blocks.3.0.dw.rbr_dense.bn.weight', 'blocks.3.0.dw.rbr_dense.bn.bias', 'blocks.3.0.dw.rbr_dense.bn.running_mean', 'blocks.3.0.dw.rbr_dense.bn.running_var', 'blocks.3.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.3.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.3.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.3.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.3.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.3.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.3.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.3.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.3.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.3.0.dw.rbr_dense_2.conv.weight', 'blocks.3.0.dw.rbr_dense_2.bn.weight', 'blocks.3.0.dw.rbr_dense_2.bn.bias', 'blocks.3.0.dw.rbr_dense_2.bn.running_mean', 'blocks.3.0.dw.rbr_dense_2.bn.running_var', 'blocks.3.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.3.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.3.0.pw.0.weight', 'blocks.3.0.pw.1.weight', 'blocks.3.0.pw.1.bias', 'blocks.3.0.pw.1.running_mean', 'blocks.3.0.pw.1.running_var', 'blocks.3.0.pw.1.num_batches_tracked', 'blocks.3.0.pw.2.needle.0.weight', 'blocks.3.0.pw.2.needle.1.weight', 'blocks.3.0.pw.2.needle.2.weight', 'blocks.3.0.pw.2.needle.3.weight', 'blocks.3.0.pw.2.needle.3.bias', 'blocks.3.0.pw.2.needle.3.running_mean', 'blocks.3.0.pw.2.needle.3.running_var', 'blocks.3.0.pw.2.needle.3.num_batches_tracked', 'blocks.3.0.conv.0.weight', 'blocks.3.0.conv.1.weight', 'blocks.3.0.conv.1.bias', 'blocks.3.0.conv.1.running_mean', 'blocks.3.0.conv.1.running_var', 'blocks.3.0.conv.1.num_batches_tracked', 'blocks.3.1.dw.rbr_dense.conv.weight', 'blocks.3.1.dw.rbr_dense.bn.weight', 'blocks.3.1.dw.rbr_dense.bn.bias', 'blocks.3.1.dw.rbr_dense.bn.running_mean', 'blocks.3.1.dw.rbr_dense.bn.running_var', 'blocks.3.1.dw.rbr_dense.bn.num_batches_tracked', 'blocks.3.1.dw.rbr_dense.conv1.needle.0.weight', 'blocks.3.1.dw.rbr_dense.conv1.needle.1.weight', 'blocks.3.1.dw.rbr_dense.conv1.needle.2.weight', 'blocks.3.1.dw.rbr_dense.conv1.needle.3.weight', 'blocks.3.1.dw.rbr_dense.conv1.needle.3.bias', 'blocks.3.1.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.3.1.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.3.1.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.3.1.dw.rbr_dense_2.conv.weight', 'blocks.3.1.dw.rbr_dense_2.bn.weight', 'blocks.3.1.dw.rbr_dense_2.bn.bias', 'blocks.3.1.dw.rbr_dense_2.bn.running_mean', 'blocks.3.1.dw.rbr_dense_2.bn.running_var', 'blocks.3.1.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.3.1.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.3.1.pw.0.weight', 'blocks.3.1.pw.1.weight', 'blocks.3.1.pw.1.bias', 'blocks.3.1.pw.1.running_mean', 'blocks.3.1.pw.1.running_var', 'blocks.3.1.pw.1.num_batches_tracked', 'blocks.3.1.pw.2.needle.0.weight', 'blocks.3.1.pw.2.needle.1.weight', 'blocks.3.1.pw.2.needle.2.weight', 'blocks.3.1.pw.2.needle.3.weight', 'blocks.3.1.pw.2.needle.3.bias', 'blocks.3.1.pw.2.needle.3.running_mean', 'blocks.3.1.pw.2.needle.3.running_var', 'blocks.3.1.pw.2.needle.3.num_batches_tracked', 'blocks.3.1.conv.0.weight', 'blocks.3.1.conv.1.weight', 'blocks.3.1.conv.1.bias', 'blocks.3.1.conv.1.running_mean', 'blocks.3.1.conv.1.running_var', 'blocks.3.1.conv.1.num_batches_tracked', 'blocks.3.2.dw.rbr_dense.conv.weight', 'blocks.3.2.dw.rbr_dense.bn.weight', 'blocks.3.2.dw.rbr_dense.bn.bias', 'blocks.3.2.dw.rbr_dense.bn.running_mean', 'blocks.3.2.dw.rbr_dense.bn.running_var', 'blocks.3.2.dw.rbr_dense.bn.num_batches_tracked', 'blocks.3.2.dw.rbr_dense.conv1.needle.0.weight', 'blocks.3.2.dw.rbr_dense.conv1.needle.1.weight', 'blocks.3.2.dw.rbr_dense.conv1.needle.2.weight', 'blocks.3.2.dw.rbr_dense.conv1.needle.3.weight', 'blocks.3.2.dw.rbr_dense.conv1.needle.3.bias', 'blocks.3.2.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.3.2.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.3.2.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.3.2.dw.rbr_dense_2.conv.weight', 'blocks.3.2.dw.rbr_dense_2.bn.weight', 'blocks.3.2.dw.rbr_dense_2.bn.bias', 'blocks.3.2.dw.rbr_dense_2.bn.running_mean', 'blocks.3.2.dw.rbr_dense_2.bn.running_var', 'blocks.3.2.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.3.weight', 
# # 'blocks.3.2.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.3.2.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.3.2.pw.0.weight', 'blocks.3.2.pw.1.weight', 'blocks.3.2.pw.1.bias', 'blocks.3.2.pw.1.running_mean', 'blocks.3.2.pw.1.running_var', 'blocks.3.2.pw.1.num_batches_tracked', 'blocks.3.2.pw.2.needle.0.weight', 'blocks.3.2.pw.2.needle.1.weight', 'blocks.3.2.pw.2.needle.2.weight', 'blocks.3.2.pw.2.needle.3.weight', 'blocks.3.2.pw.2.needle.3.bias', 'blocks.3.2.pw.2.needle.3.running_mean', 'blocks.3.2.pw.2.needle.3.running_var', 'blocks.3.2.pw.2.needle.3.num_batches_tracked', 'blocks.3.2.conv.0.weight', 'blocks.3.2.conv.1.weight', 'blocks.3.2.conv.1.bias', 'blocks.3.2.conv.1.running_mean', 'blocks.3.2.conv.1.running_var', 'blocks.3.2.conv.1.num_batches_tracked', 'blocks.3.3.dw.rbr_dense.conv.weight', 'blocks.3.3.dw.rbr_dense.bn.weight', 'blocks.3.3.dw.rbr_dense.bn.bias', 'blocks.3.3.dw.rbr_dense.bn.running_mean', 'blocks.3.3.dw.rbr_dense.bn.running_var', 'blocks.3.3.dw.rbr_dense.bn.num_batches_tracked', 'blocks.3.3.dw.rbr_dense.conv1.needle.0.weight', 'blocks.3.3.dw.rbr_dense.conv1.needle.1.weight', 'blocks.3.3.dw.rbr_dense.conv1.needle.2.weight', 'blocks.3.3.dw.rbr_dense.conv1.needle.3.weight', 'blocks.3.3.dw.rbr_dense.conv1.needle.3.bias', 'blocks.3.3.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.3.3.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.3.3.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.3.3.dw.rbr_dense_2.conv.weight', 'blocks.3.3.dw.rbr_dense_2.bn.weight', 'blocks.3.3.dw.rbr_dense_2.bn.bias', 'blocks.3.3.dw.rbr_dense_2.bn.running_mean', 'blocks.3.3.dw.rbr_dense_2.bn.running_var', 'blocks.3.3.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.3.3.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.3.3.pw.0.weight', 'blocks.3.3.pw.1.weight', 'blocks.3.3.pw.1.bias', 'blocks.3.3.pw.1.running_mean', 'blocks.3.3.pw.1.running_var', 'blocks.3.3.pw.1.num_batches_tracked', 'blocks.3.3.pw.2.needle.0.weight', 'blocks.3.3.pw.2.needle.1.weight', 'blocks.3.3.pw.2.needle.2.weight', 'blocks.3.3.pw.2.needle.3.weight', 'blocks.3.3.pw.2.needle.3.bias', 'blocks.3.3.pw.2.needle.3.running_mean', 'blocks.3.3.pw.2.needle.3.running_var', 'blocks.3.3.pw.2.needle.3.num_batches_tracked', 'blocks.3.3.conv.0.weight', 'blocks.3.3.conv.1.weight', 'blocks.3.3.conv.1.bias', 'blocks.3.3.conv.1.running_mean', 'blocks.3.3.conv.1.running_var', 'blocks.3.3.conv.1.num_batches_tracked', 'blocks.4.0.dw.rbr_dense.conv.weight', 'blocks.4.0.dw.rbr_dense.bn.weight', 'blocks.4.0.dw.rbr_dense.bn.bias', 'blocks.4.0.dw.rbr_dense.bn.running_mean', 'blocks.4.0.dw.rbr_dense.bn.running_var', 'blocks.4.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.4.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.4.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.4.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.4.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.4.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.4.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.4.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.4.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.4.0.dw.rbr_dense_2.conv.weight', 'blocks.4.0.dw.rbr_dense_2.bn.weight', 'blocks.4.0.dw.rbr_dense_2.bn.bias', 'blocks.4.0.dw.rbr_dense_2.bn.running_mean', 'blocks.4.0.dw.rbr_dense_2.bn.running_var', 'blocks.4.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.4.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.4.0.pw.0.weight', 'blocks.4.0.pw.1.weight', 'blocks.4.0.pw.1.bias', 'blocks.4.0.pw.1.running_mean', 'blocks.4.0.pw.1.running_var', 'blocks.4.0.pw.1.num_batches_tracked', 'blocks.4.0.pw.2.needle.0.weight', 'blocks.4.0.pw.2.needle.1.weight', 'blocks.4.0.pw.2.needle.2.weight', 'blocks.4.0.pw.2.needle.3.weight', 'blocks.4.0.pw.2.needle.3.bias', 'blocks.4.0.pw.2.needle.3.running_mean', 'blocks.4.0.pw.2.needle.3.running_var', 'blocks.4.0.pw.2.needle.3.num_batches_tracked', 'blocks.4.0.conv.0.weight', 'blocks.4.0.conv.1.weight', 'blocks.4.0.conv.1.bias', 'blocks.4.0.conv.1.running_mean', 'blocks.4.0.conv.1.running_var', 'blocks.4.0.conv.1.num_batches_tracked', 'blocks.4.1.dw.rbr_dense.conv.weight', 'blocks.4.1.dw.rbr_dense.bn.weight', 'blocks.4.1.dw.rbr_dense.bn.bias', 'blocks.4.1.dw.rbr_dense.bn.running_mean', 'blocks.4.1.dw.rbr_dense.bn.running_var', 'blocks.4.1.dw.rbr_dense.bn.num_batches_tracked', 'blocks.4.1.dw.rbr_dense.conv1.needle.0.weight', 'blocks.4.1.dw.rbr_dense.conv1.needle.1.weight', 'blocks.4.1.dw.rbr_dense.conv1.needle.2.weight', 'blocks.4.1.dw.rbr_dense.conv1.needle.3.weight', 'blocks.4.1.dw.rbr_dense.conv1.needle.3.bias', 'blocks.4.1.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.4.1.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.4.1.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.4.1.dw.rbr_dense_2.conv.weight', 'blocks.4.1.dw.rbr_dense_2.bn.weight', 'blocks.4.1.dw.rbr_dense_2.bn.bias', 'blocks.4.1.dw.rbr_dense_2.bn.running_mean', 'blocks.4.1.dw.rbr_dense_2.bn.running_var', 'blocks.4.1.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.4.1.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.4.1.pw.0.weight', 'blocks.4.1.pw.1.weight', 'blocks.4.1.pw.1.bias', 'blocks.4.1.pw.1.running_mean', 'blocks.4.1.pw.1.running_var', 'blocks.4.1.pw.1.num_batches_tracked', 'blocks.4.1.pw.2.needle.0.weight', 'blocks.4.1.pw.2.needle.1.weight', 'blocks.4.1.pw.2.needle.2.weight', 'blocks.4.1.pw.2.needle.3.weight', 'blocks.4.1.pw.2.needle.3.bias', 'blocks.4.1.pw.2.needle.3.running_mean', 'blocks.4.1.pw.2.needle.3.running_var', 'blocks.4.1.pw.2.needle.3.num_batches_tracked', 'blocks.4.1.conv.0.weight', 'blocks.4.1.conv.1.weight', 'blocks.4.1.conv.1.bias', 'blocks.4.1.conv.1.running_mean', 'blocks.4.1.conv.1.running_var', 'blocks.4.1.conv.1.num_batches_tracked', 'blocks.4.2.dw.rbr_dense.conv.weight', 'blocks.4.2.dw.rbr_dense.bn.weight', 'blocks.4.2.dw.rbr_dense.bn.bias', 'blocks.4.2.dw.rbr_dense.bn.running_mean', 'blocks.4.2.dw.rbr_dense.bn.running_var', 'blocks.4.2.dw.rbr_dense.bn.num_batches_tracked', 'blocks.4.2.dw.rbr_dense.conv1.needle.0.weight', 'blocks.4.2.dw.rbr_dense.conv1.needle.1.weight', 'blocks.4.2.dw.rbr_dense.conv1.needle.2.weight', 'blocks.4.2.dw.rbr_dense.conv1.needle.3.weight', 'blocks.4.2.dw.rbr_dense.conv1.needle.3.bias', 'blocks.4.2.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.4.2.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.4.2.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.4.2.dw.rbr_dense_2.conv.weight', 'blocks.4.2.dw.rbr_dense_2.bn.weight', 'blocks.4.2.dw.rbr_dense_2.bn.bias', 'blocks.4.2.dw.rbr_dense_2.bn.running_mean', 'blocks.4.2.dw.rbr_dense_2.bn.running_var', 'blocks.4.2.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.4.2.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.4.2.pw.0.weight', 'blocks.4.2.pw.1.weight', 'blocks.4.2.pw.1.bias', 'blocks.4.2.pw.1.running_mean', 'blocks.4.2.pw.1.running_var', 'blocks.4.2.pw.1.num_batches_tracked', 'blocks.4.2.pw.2.needle.0.weight', 'blocks.4.2.pw.2.needle.1.weight', 'blocks.4.2.pw.2.needle.2.weight', 'blocks.4.2.pw.2.needle.3.weight', 'blocks.4.2.pw.2.needle.3.bias', 'blocks.4.2.pw.2.needle.3.running_mean', 'blocks.4.2.pw.2.needle.3.running_var', 'blocks.4.2.pw.2.needle.3.num_batches_tracked', 'blocks.4.2.conv.0.weight', 'blocks.4.2.conv.1.weight', 'blocks.4.2.conv.1.bias', 'blocks.4.2.conv.1.running_mean', 'blocks.4.2.conv.1.running_var', 'blocks.4.2.conv.1.num_batches_tracked', 'blocks.5.0.dw.rbr_dense.conv.weight', 'blocks.5.0.dw.rbr_dense.bn.weight', 'blocks.5.0.dw.rbr_dense.bn.bias', 'blocks.5.0.dw.rbr_dense.bn.running_mean', 'blocks.5.0.dw.rbr_dense.bn.running_var', 'blocks.5.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.5.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.5.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.5.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.5.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.5.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.5.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.5.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.5.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.5.0.dw.rbr_dense_2.conv.weight', 'blocks.5.0.dw.rbr_dense_2.bn.weight', 'blocks.5.0.dw.rbr_dense_2.bn.bias', 'blocks.5.0.dw.rbr_dense_2.bn.running_mean', 'blocks.5.0.dw.rbr_dense_2.bn.running_var', 'blocks.5.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.5.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.5.0.dw.rbr_dense_2.conv1.needle.1.weight',
# #  'blocks.5.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.5.0.dw.rbr_dense_2.conv1.needle.3.weight', 
# # 'blocks.5.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.5.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.5.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.5.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.5.0.pw.0.weight', 'blocks.5.0.pw.1.weight', 'blocks.5.0.pw.1.bias', 'blocks.5.0.pw.1.running_mean', 'blocks.5.0.pw.1.running_var', 'blocks.5.0.pw.1.num_batches_tracked', 'blocks.5.0.pw.2.needle.0.weight', 'blocks.5.0.pw.2.needle.1.weight', 'blocks.5.0.pw.2.needle.2.weight', 'blocks.5.0.pw.2.needle.3.weight', 'blocks.5.0.pw.2.needle.3.bias', 'blocks.5.0.pw.2.needle.3.running_mean', 'blocks.5.0.pw.2.needle.3.running_var', 'blocks.5.0.pw.2.needle.3.num_batches_tracked', 'blocks.5.0.conv.0.weight', 'blocks.5.0.conv.1.weight', 'blocks.5.0.conv.1.bias', 'blocks.5.0.conv.1.running_mean', 'blocks.5.0.conv.1.running_var', 'blocks.5.0.conv.1.num_batches_tracked', 'blocks.5.1.dw.rbr_dense.conv.weight', 'blocks.5.1.dw.rbr_dense.bn.weight', 'blocks.5.1.dw.rbr_dense.bn.bias', 'blocks.5.1.dw.rbr_dense.bn.running_mean', 'blocks.5.1.dw.rbr_dense.bn.running_var', 'blocks.5.1.dw.rbr_dense.bn.num_batches_tracked', 'blocks.5.1.dw.rbr_dense.conv1.needle.0.weight', 'blocks.5.1.dw.rbr_dense.conv1.needle.1.weight', 'blocks.5.1.dw.rbr_dense.conv1.needle.2.weight', 'blocks.5.1.dw.rbr_dense.conv1.needle.3.weight', 'blocks.5.1.dw.rbr_dense.conv1.needle.3.bias', 'blocks.5.1.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.5.1.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.5.1.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.5.1.dw.rbr_dense_2.conv.weight', 'blocks.5.1.dw.rbr_dense_2.bn.weight', 'blocks.5.1.dw.rbr_dense_2.bn.bias', 'blocks.5.1.dw.rbr_dense_2.bn.running_mean', 'blocks.5.1.dw.rbr_dense_2.bn.running_var', 'blocks.5.1.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.5.1.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.5.1.pw.0.weight', 'blocks.5.1.pw.1.weight', 'blocks.5.1.pw.1.bias', 'blocks.5.1.pw.1.running_mean', 'blocks.5.1.pw.1.running_var', 'blocks.5.1.pw.1.num_batches_tracked', 'blocks.5.1.pw.2.needle.0.weight', 'blocks.5.1.pw.2.needle.1.weight', 'blocks.5.1.pw.2.needle.2.weight', 'blocks.5.1.pw.2.needle.3.weight', 'blocks.5.1.pw.2.needle.3.bias', 'blocks.5.1.pw.2.needle.3.running_mean', 'blocks.5.1.pw.2.needle.3.running_var', 'blocks.5.1.pw.2.needle.3.num_batches_tracked', 'blocks.5.1.conv.0.weight', 'blocks.5.1.conv.1.weight', 'blocks.5.1.conv.1.bias', 'blocks.5.1.conv.1.running_mean', 'blocks.5.1.conv.1.running_var', 'blocks.5.1.conv.1.num_batches_tracked', 'blocks.5.2.dw.rbr_dense.conv.weight', 'blocks.5.2.dw.rbr_dense.bn.weight', 'blocks.5.2.dw.rbr_dense.bn.bias', 'blocks.5.2.dw.rbr_dense.bn.running_mean', 'blocks.5.2.dw.rbr_dense.bn.running_var', 'blocks.5.2.dw.rbr_dense.bn.num_batches_tracked', 'blocks.5.2.dw.rbr_dense.conv1.needle.0.weight', 'blocks.5.2.dw.rbr_dense.conv1.needle.1.weight', 'blocks.5.2.dw.rbr_dense.conv1.needle.2.weight', 'blocks.5.2.dw.rbr_dense.conv1.needle.3.weight', 'blocks.5.2.dw.rbr_dense.conv1.needle.3.bias', 'blocks.5.2.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.5.2.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.5.2.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.5.2.dw.rbr_dense_2.conv.weight', 'blocks.5.2.dw.rbr_dense_2.bn.weight', 'blocks.5.2.dw.rbr_dense_2.bn.bias', 'blocks.5.2.dw.rbr_dense_2.bn.running_mean', 'blocks.5.2.dw.rbr_dense_2.bn.running_var', 'blocks.5.2.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.5.2.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.5.2.pw.0.weight', 'blocks.5.2.pw.1.weight', 'blocks.5.2.pw.1.bias', 'blocks.5.2.pw.1.running_mean', 'blocks.5.2.pw.1.running_var', 'blocks.5.2.pw.1.num_batches_tracked', 'blocks.5.2.pw.2.needle.0.weight', 'blocks.5.2.pw.2.needle.1.weight', 'blocks.5.2.pw.2.needle.2.weight', 'blocks.5.2.pw.2.needle.3.weight', 'blocks.5.2.pw.2.needle.3.bias', 'blocks.5.2.pw.2.needle.3.running_mean', 'blocks.5.2.pw.2.needle.3.running_var', 'blocks.5.2.pw.2.needle.3.num_batches_tracked', 'blocks.5.2.conv.0.weight', 'blocks.5.2.conv.1.weight', 'blocks.5.2.conv.1.bias', 'blocks.5.2.conv.1.running_mean', 'blocks.5.2.conv.1.running_var', 'blocks.5.2.conv.1.num_batches_tracked', 'blocks.6.0.dw.rbr_dense.conv.weight', 'blocks.6.0.dw.rbr_dense.bn.weight', 'blocks.6.0.dw.rbr_dense.bn.bias', 'blocks.6.0.dw.rbr_dense.bn.running_mean', 'blocks.6.0.dw.rbr_dense.bn.running_var', 'blocks.6.0.dw.rbr_dense.bn.num_batches_tracked', 'blocks.6.0.dw.rbr_dense.conv1.needle.0.weight', 'blocks.6.0.dw.rbr_dense.conv1.needle.1.weight', 'blocks.6.0.dw.rbr_dense.conv1.needle.2.weight', 'blocks.6.0.dw.rbr_dense.conv1.needle.3.weight', 'blocks.6.0.dw.rbr_dense.conv1.needle.3.bias', 'blocks.6.0.dw.rbr_dense.conv1.needle.3.running_mean', 'blocks.6.0.dw.rbr_dense.conv1.needle.3.running_var', 'blocks.6.0.dw.rbr_dense.conv1.needle.3.num_batches_tracked', 'blocks.6.0.dw.rbr_dense_2.conv.weight', 'blocks.6.0.dw.rbr_dense_2.bn.weight', 'blocks.6.0.dw.rbr_dense_2.bn.bias', 'blocks.6.0.dw.rbr_dense_2.bn.running_mean', 'blocks.6.0.dw.rbr_dense_2.bn.running_var', 'blocks.6.0.dw.rbr_dense_2.bn.num_batches_tracked', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.0.weight', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.1.weight', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.2.weight', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.3.weight', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.3.bias', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.3.running_mean', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.3.running_var', 'blocks.6.0.dw.rbr_dense_2.conv1.needle.3.num_batches_tracked', 'blocks.6.0.pw.0.weight', 'blocks.6.0.pw.1.weight', 'blocks.6.0.pw.1.bias', 'blocks.6.0.pw.1.running_mean', 'blocks.6.0.pw.1.running_var', 'blocks.6.0.pw.1.num_batches_tracked', 'blocks.6.0.pw.2.needle.0.weight', 'blocks.6.0.pw.2.needle.1.weight', 'blocks.6.0.pw.2.needle.2.weight', 'blocks.6.0.pw.2.needle.3.weight', 'blocks.6.0.pw.2.needle.3.bias', 'blocks.6.0.pw.2.needle.3.running_mean', 'blocks.6.0.pw.2.needle.3.running_var', 'blocks.6.0.pw.2.needle.3.num_batches_tracked', 'blocks.6.0.conv.0.weight', 'blocks.6.0.conv.1.weight', 'blocks.6.0.conv.1.bias', 'blocks.6.0.conv.1.running_mean', 'blocks.6.0.conv.1.running_var', 'blocks.6.0.conv.1.num_batches_tracked', 'classifier.classifier.l.weight', 'classifier.classifier.l.bias', 'classifier.classifier_dist.l.weight', 'classifier.classifier_dist.l.bias'])

# import numpy as np

# # 输入 4x4 图像矩阵
# image = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ])

# # 打印原始矩阵
# print("原始矩阵：")
# print(image)

# # 对矩阵进行二维FFT变换
# fft_result = np.fft.fft2(image)

# # 打印二维FFT的结果（复数形式）
# print("\n二维 FFT 结果 (复数形式)：")
# print(fft_result)

# # 打印FFT的频谱幅值
# fft_magnitude = np.abs(fft_result)
# print("\nFFT 频谱幅值：")
# print(fft_magnitude)

# # 打印FFT的相位
# fft_phase = np.angle(fft_result)
# print("\nFFT 相位：")
# print(fft_phase)

