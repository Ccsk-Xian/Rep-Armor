import time
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
# 1x1+3x3+1x1

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module('conv_1', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=3, stride=1, padding=1,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepNeedleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,expand_rate = 3,skip=True):
        super(RepNeedleBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

     
        # 不能引入bias，所以只能是identity
        skip =False
        if skip:
            self.skip = nn.Identity()
        else:
            self.skip = None
        if self.deploy:
            print(2423)
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                      padding=1, dilation=dilation, groups=self.groups, bias=False, padding_mode=padding_mode)
        else:
            self.needle = nn.Sequential(
                nn.Conv2d(in_channels,in_channels*expand_rate,1,1,0,bias=False,groups=groups),
                nn.Conv2d(in_channels*expand_rate,in_channels*expand_rate,3,1,1,bias=False,groups=groups),
                nn.Conv2d(in_channels*expand_rate,out_channels,1,1,0,bias=False,groups=groups),
            )
            # self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,bias=False)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            # print('222true')
            print('**'+str(self.rbr_reparam.weight.shape))
            return self.rbr_reparam(inputs)
            
        if self.skip != None:
            print('111true')
            return self.needle(inputs)+self.skip(inputs)
            
            # return self.needle(inputs)
        else:
            print('333true')
            return self.needle(inputs)
            

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel_0 = branch[0].weight
            kernel_1 = branch[1].weight
            # kernel_2 = branch[2].weight
            d = torch.einsum('ijkl,jskl->iskl',kernel_1,kernel_0)
            kernel_f = torch.einsum('ijkl,jskl->iskl',kernel_2,d)
            
        return kernel_f

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        if self.needle is None:
            return 0, 0
        if isinstance(self.needle, nn.Sequential):
            kernel_0 = self.needle[0].weight
            kernel_1 = self.needle[1].weight
            kernel_2 = self.needle[2].weight
            print(kernel_0.shape)
            print(kernel_1.shape)
            # print(kernel_1.view(kernel_2.shape[1],kernel_1.shape[0],1,1).shape)
            print(kernel_2.shape)
            # d = torch.einsum('ijkl,jskl->iskl',kernel_1,kernel_0)
            # 

            # d = torch.einsum('ijkl,jskl->iskl',kernel_2,kernel_1)
            # kernel_f = torch.einsum('ijkl,jskl->iskl',d,kernel_0)
            # torch.flip((F.conv2d(kernel_0.transpose(0,1),kernel_1,stride=1,padding=2,groups=self.groups).transpose(0,1)),(2,3))
            # torch.flip((F.conv2d(d.transpose(0,1),kernel_2,stride=1,padding=0,groups=self.groups).transpose(0,1)),(2,3))
            
            # d =  F.conv2d(kernel_1, kernel_0.permute(1, 0, 2, 3))  
            # kernel_f = F.conv2d(kernel_2, d.permute(1, 0, 2, 3)) 
            
            d=torch.flip((F.conv2d(kernel_0.transpose(0,1),kernel_1,stride=1,padding=2,groups=self.groups).transpose(0,1)),(2,3))
            # print('d'+str(d.shape))
            # kernel_f=torch.flip((F.conv2d(d.transpose(0,1),kernel_2,stride=1,padding=0,groups=self.groups).transpose(0,1)),(2,3))
            kernel_f = (F.conv2d(d.transpose(0,1),kernel_2,stride=1,padding=0,groups=self.groups).transpose(0,1))
            # print('k'+str(kernel_f.shape))
            # d = torch.einsum('ijkl,jskl->iskl',kernel_2,kernel_1)
            # kernel_f = torch.einsum('ijkl,jskl->iskl',d,kernel_0)
            # kernel_f=torch.flip((F.conv2d(kernel_0.transpose(0,1),d,stride=1,padding=0,groups=1).transpose(0,1)),(2,3))
            # print(d.shape)
            # =torch.flip((F.conv2d(d.transpose(0,1),kernel_2,stride=1,padding=0,groups=1).transpose(0,1)),(2,3))
            print(kernel_f.shape)

            # if not hasattr(self, 'id_tensor'):
            #     input_dim = self.in_channels // self.groups
            #     kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            #     for i in range(self.in_channels):
            #         kernel_value[i, i % input_dim, 0, 0] = 1
            #     self.id_tensor = torch.from_numpy(kernel_value).to(self.needle[0].weight.device)
            # kernel = self.id_tensor
            # kernel_f = kernel+kernel_f
        self.rbr_reparam = nn.Conv2d(in_channels=self.needle[0].in_channels,
                                     out_channels=self.needle[-1].out_channels,
                                     kernel_size=3, stride=1,
                                     padding=1, dilation=self.needle[0].dilation,
                                     groups=self.needle[0].groups, bias=False)
        self.rbr_reparam.weight.data = kernel_f
        for para in self.parameters():
            para.detach_()
        self.__delattr__('needle')
        if hasattr(self, 'skip'):
            self.__delattr__('skip')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class TestModel(nn.Module):
    def __init__(self, in_channels=64, out_channels=64,groups=1):
        super(TestModel, self).__init__()
        self.groups = groups
        self.needle = RepNeedleBlock(in_channels=in_channels, out_channels=out_channels,groups=groups)
        # self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,dilation=1,groups=groups,bias=False)
        # self.needle = RepNeedleBlock(in_channels=out_channels, out_channels=out_channels,groups=groups)
    def forward(self,inputs):
        # print(self.conv.weight.shape)
        # 
        if hasattr(self, 'rbr_reparam1'):
            print('156true')
            return self.rbr_reparam1(inputs)
        else:
            # return self.conv(self.needle(inputs))
            # return self.needle(self.conv(inputs))
            return self.needle(inputs)
    
    def switch_to_deploy1(self):
        if hasattr(self, 'rbr_reparam1'):
            return
        if self.needle is None:
            return 0, 0
        if isinstance(self.needle, nn.Module):
            kernel_0 = self.needle.rbr_reparam.weight
            kernel_1 = self.conv.weight
            
            print(kernel_0.shape)
            print(kernel_1.shape)
            # print(kernel_1.view(kernel_2.shape[1],kernel_1.shape[0],1,1).shape)
            # print(kernel_2.shape)
            # d = torch.einsum('ijkl,jskl->iskl',kernel_1,kernel_0)
            # kernel_f = torch.einsum('ijkl,jskl->iskl',kernel_2,d)

            # d = torch.einsum('ijkl,jskl->iskl',kernel_2,kernel_1)
            # kernel_f1 = torch.einsum('ijkl,jskl->iskl',kernel_1,kernel_0)
            # print(kernel_f1)
            # kernel_f = torch.einsum('ijkl,jskl->iskl',kernel_0,kernel_1)
            # 先needle后conv
            # kernel_f=torch.flip((F.conv2d(kernel_0.transpose(0,1),kernel_1,stride=1,padding=2,groups=self.groups).transpose(0,1)),(2,3))
            kernel_f=(F.conv2d(kernel_1.transpose(0,1),kernel_0,stride=1,padding=0,groups=self.groups).transpose(0,1))
            # print(kernel_f)
            # print(torch.flip(kernel_f,(2,3)))
            # print(torch.flip(kernel_f,(2,3))==kernel_f1)
            # print('**')
            # print(kernel_f.shape)
            # kernel_f=F.conv2d(d.transpose(0,1),kernel_2,stride=1,padding=0,groups=self.groups).transpose(0,1)

            # if not hasattr(self, 'id_tensor'):
            #     input_dim = self.in_channels // self.groups
            #     kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            #     for i in range(self.in_channels):
            #         kernel_value[i, i % input_dim, 0, 0] = 1
            #     self.id_tensor = torch.from_numpy(kernel_value).to(self.needle[0].weight.device)
            # kernel = self.id_tensor
            # kernel_f = kernel+kernel_f
        self.rbr_reparam1 = nn.Conv2d(in_channels=self.conv.in_channels,
                                     out_channels=self.conv.out_channels,
                                     kernel_size=3, stride=1,
                                     padding=1, dilation=self.conv.dilation,
                                     groups=self.conv.groups, bias=False)
        self.rbr_reparam1.weight.data = kernel_f
        for para in self.parameters():
            para.detach_()
        self.__delattr__('needle')
        self.__delattr__('conv')
        if hasattr(self, 'skip'):
            self.__delattr__('skip')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

def main():
    f1 = torch.randn(1, 64, 64, 64)
    # block = RepNeedleBlock(in_channels=64, out_channels=64)
    block = TestModel(in_channels=64, out_channels=64,groups=1)
    # print(block)
    # print(len(block.needle))
    # print(block.needle[0].weight)
    block.eval()
    with torch.no_grad():
        output1 = block(f1)
        start_time = time.time()
        for _ in range(1):
            block(f1)
        print(f"consume time: {time.time() - start_time}")
        total_params1 = sum(p.numel() for p in block.parameters())
        print('total_params_before_merge:'+str(total_params1))
        # re-parameterization
        # block.switch_to_deploy()
        block.needle.switch_to_deploy()
        # block.switch_to_deploy1()
        output2 = block(f1)
        start_time = time.time()
        for _ in range(1):
            block(f1)
        print(f"consume time: {time.time() - start_time}")
        total_params2 = sum(p.numel() for p in block.parameters())
        print('total_params_after_merge:'+str(total_params2))
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
        print("convert module has been tested, and the inference results, before and after merging, are equal!")


if __name__ == '__main__':
    main()
