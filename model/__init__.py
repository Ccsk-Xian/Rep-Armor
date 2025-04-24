from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50,ResNet18
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .mobilenetv2_9 import mobile_half_9
from .mobilenetv3 import mobilev3_half
from .mobilenetContainer import mobilecontainer
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
# 起始下采样层使用rep，扩大感受野范围
from .mobilevit import mobile_vit_small_in7,mobile_vit_xx_small_in7,mobile_vit_x_small_in7,mobile_vit_tiny_in7
# 无修改的mobilevit
from .mobilevit_init import mobile_vit_small_init,mobile_vit_xx_small_init,mobile_vit_x_small_init,mobile_vit_tiny_init
from .mobilevit_in_all import mobile_vit_tiny
from .mobilenet_ import mobile_vit_tiny_likevit
from .mobilenet_2 import mobile_vit_tiny_likevitpp
from .mobilevit_novit import mobile_vit_tiny_novit
from .mobilevit_best import mobile_vit_xx_small_best,mobile_vit_tiny_best
from .mobile_novit_test1 import mobile_vit_tiny_novit_test1
from .mobile_novit_test2 import mobile_vit_tiny_novit_test2
from .mobile_novit_test3 import mobile_vit_tiny_novit_test3
from .mobile_novit_test4 import mobile_vit_tiny_novit_test4
from .mobile_novit_test5 import mobile_vit_tiny_novit_test5
from .mobile_novit_test6 import mobile_vit_tiny_novit_test6
from .mobile_novit_test0 import mobile_vit_tiny_novit_test0
from .mobile_novit_unireplknet import mobile_vit_tiny_dilatedblock_5
from .mobile_novit_unireplknet_1 import mobile_vit_tiny_dilatedblock_5_1
from .mobile_novit_unireplknet_2 import mobile_vit_tiny_dilatedblock_5_2
from .mobile_novit_unireplknet_3 import mobile_vit_tiny_dilatedblock_5_3
from .mobile_novit_unireplknet_4 import mobile_vit_tiny_dilatedblock_5_4
from .mobile_novit_unireplknet_5 import mobile_vit_tiny_dilatedblock_5_5
from .mobile_novit_unireplknet_5_1 import mobile_vit_tiny_dilatedblock_5_5_1
from .mobile_novit_unireplknet_7 import mobile_vit_tiny_dilatedblock_7
from .mobile_novit_unireplknet_9 import mobile_vit_tiny_dilatedblock_9
from .mobile_novit_unireplknet_noweight import mobile_vit_tiny_dilatedblock_5_noweight
from .SwiftFomer import SwiftFormer_XXS
from .EdgexNet import edgenext_xxx_small
from .RepVit import repvit_m0_6
from .RepInfiniteUniVit import repvit_m0_6_infiuni
from .RepInfiniteUniVit_initial import repvit_m0_6_uni_initial
from .RepInfiniteUniVit_ours import repvit_m0_6_uni_ours
from .mobilenetv2stem1_1_1 import mobile_half_1_1_1
from .mobilenetv2stem1_1_1_1 import mobile_half_1_1_1_1
from .mobilenetv2stem1_1_1_2 import mobile_half_1_1_1_2
from .mobilenetv2stem1_1_2 import mobile_half_1_1_2
from .mobilenetv2stem1_1_3 import mobile_half_1_1_3
from .mobilenetv2stem1_1_4 import mobile_half_1_1_4
from .mobilenetv2stem1_1_5 import mobile_half_1_1_5
from .mobilenetv2stem1_1_5_1 import mobile_half_1_1_5_1
from .mobilenetv2stem1_1_5_2 import mobile_half_1_1_5_2
from .mobilenetv2stem1_1_5_3 import mobile_half_1_1_5_3
from .mobilenetv2stem1_1_5_3_1 import mobile_half_1_1_5_3_1
from .mobilenetv2stem1_1_5_4 import mobile_half_1_1_5_4
from .mobilenetv2stem1_1_7 import mobile_half_1_1_7
from .mobilenetv2stem1_1_8 import mobile_half_1_1_8
from .mobilenetv2stem1_1_8_1 import mobile_half_1_1_8_1
from .mobilenetv2stem1_1_8_2 import mobile_half_1_1_8_2
from .mobilenetv2stem1_1_8_3 import mobile_half_1_1_8_3
from .mobilenetv2stem1_1_8_4 import mobile_half_1_1_8_4
from .mobilenetv2stem1_1_8_5 import mobile_half_1_1_8_5
from .mobilenetv2stem1_1_8_6 import mobile_half_1_1_8_6
from .mobilenetv2stem1_1_8_7 import mobile_half_1_1_8_7
from .mobilenetv2stem1_1_9 import mobile_half_1_1_9
from .mobilenetv2stem1_1_10 import mobile_half_1_1_10
from .mobilenetv2stem1_1_11 import mobile_half_1_1_11
from .mobilenetv2stem1_1_12 import mobile_half_1_1_12
from .mobilenetv2stem1_1_12_1 import mobile_half_1_1_12_1
from .mobilenetv2stem1_2_1 import mobile_half_1_2_1
from .mobilenetv2stem1_2_1_1 import mobile_half_1_2_1_1
from .mobilenetv2stem1_2_2 import mobile_half_1_2_2
from .mobilenetv2stem1_2_2_1 import mobile_half_1_2_2_1
from .mobilenetv2stem1_2_3 import mobile_half_1_2_3
from .mobilenetv2stem1_2_4 import mobile_half_1_2_4
from .mobilenetv2stem1_3_1 import mobile_half_1_3_1
from .mobilenetv2down2_1_1 import mobile_half_2_1_1
from .mobilenetv2down2_1_2 import mobile_half_2_1_2
from .mobilenetv2down2_1_3 import mobile_half_2_1_3
from .mobilenetv2down2_1_4 import mobile_half_2_1_4
from .mobilenetv2down2_1_5 import mobile_half_2_1_5
from .mobilenetv2down2_1_6 import mobile_half_2_1_6
from .mobilenetv2down2_1_1_1 import mobile_half_2_1_1_1
from .mobilenetv2class3_1_1 import mobile_half_3_1_1
from .mobilenetv2class3_1_2 import mobile_half_3_1_2
from .mobilenetv2main4_1_1 import mobile_half_4_1_1
from .mobilenetv2main4_1_2 import mobile_half_4_1_2
from .mobilenetv2main4_1_2_1 import mobile_half_4_1_2_1
from .mobilenetv2main4_1_2_2 import mobile_half_4_1_2_2
from .mobilenetv2main4_1_3 import mobile_half_4_1_3
from .mobilenetv2main4_1_3_1 import mobile_half_4_1_3_1
from .mobilenetv2main4_1_3_2 import mobile_half_4_1_3_2
from .mobilenetv2main4_1_3_3 import mobile_half_4_1_3_3
from .mobilenetv2main4_1_4 import mobile_half_4_1_4
from .mobilenetv2main4_1_5 import mobile_half_4_1_5
from .mobilenetv2main4_1_6 import mobile_half_4_1_6
from .mobilenetv2main4_1_7 import mobile_half_4_1_7
from .mobilenetv2all_5_1_1 import mobile_half_5_1_1
from .mobilenetv2all_5_1_1_1 import mobile_half_5_1_1_1
from .mobilenetv2all_5_1_1_2 import mobile_half_5_1_1_2
from .mobilenetv2all_5_1_1_3 import mobile_half_5_1_1_3
from .mobilenetv2all_5_1_1_4 import mobile_half_5_1_1_4
from .mobilenetv2percent import mobile_half_percent
from .mobilenetv2dim6_1_1_1 import mobile_half_6_1_1_1
from .mobilenetv2dim6_1_1_2 import mobile_half_6_1_1_2
from .mobilenetv2dim6_1_1_2_1 import mobile_half_6_1_1_2_1
from .mobilenetv2dim6_1_2_1 import mobile_half_6_1_2_1
from .mobilenetv2smallres_7_1_1 import mobile_half_7_1_1
from .mobilenetv2smallres_7_1_2 import mobile_half_7_1_2
from .mobilenetv2smallres_7_1_1_1 import mobile_half_7_1_1_1
from .mobilenetv2smallres_7_1_2_1 import mobile_half_7_1_2_1
from .mobilemeta import mobilemetanet
from .mobilemeta_1 import mobilemetanet_1
from .mobilemeta_2 import mobilemetanet_2
from .RepTinyNet import RepTinynet
from .RepTinyNet1 import RepTinynet1
from .RepTinyNet2 import RepTinynet2
from .RepTinyNet3 import RepTinynet3
from .RepTinyNet4 import RepTinynet4
from .RepTinyNet5 import RepTinynet5
from .RepTinyNet6 import RepTinynet6
from .RepTinyNet7 import RepTinynet7
from .RepTinyNet8 import RepTinynet8
from .RepTinyNet9 import RepTinynet9
from .RepTinyNet10 import RepTinynet10
from .RepTinyNet11 import RepTinynet11
from .RepTinyNet12 import RepTinynet12
from .RepTinyNet13 import RepTinynet13
from .RepTinyNet14 import RepTinynet14
from .RepTinyNet15 import RepTinynet15
from .RepTinyNet16 import RepTinynet16
from .RepTinyNet17 import RepTinynet17
from .RepTinyNet18 import RepTinynet18
from .RepTinyNet19 import RepTinynet19
from .RepTinyNet20 import RepTinynet20
from .RepTinyNet21 import RepTinynet21
from .RepTinyNet22 import RepTinynet22
from .RepTinyNet23 import RepTinynet23
from .RepTinyNet24 import RepTinynet24
from .RepTinyNet25 import RepTinynet25
from .mculike import mcunetlike
from .mobilenetv2_class import mobile_half_class
from .mobilenetv2_class_2 import mobile_half_class_2
from .mobilenetv2_class_3 import mobile_half_class_3
from .repvgg import repvggnet
# from Mcunet.Mcunet.model_zoo import McuNetv1 as MCUnetv1
from .mcunetv1 import MCUnetv1
from .Contrast1 import ContrastNet1
from .Contrast2 import ContrastNet2
from .Contrast3 import ContrastNet3
from .Contrast4 import ContrastNet4
from .Contrast5 import ContrastNet5
from .Contrast6 import ContrastNet6
from .Contrast7 import ContrastNet7
from .Contrast8 import ContrastNet8
from .Contrast9 import ContrastNet9
from .Contrast10 import ContrastNet10
from .Contrast11 import ContrastNet11
from .Contrast12 import ContrastNet12
from .final import finalNet
from .squeezenet import squeezenet1
from .mobilenetv2_aux import mobile_half_aux
from .aug import mobile_half_aux_1
from .expand import mobilenetv2_expand
model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'aug':mobile_half_aux_1,
    'expand':mobilenetv2_expand,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'ResNet18':ResNet18,
    'mobile_half_aux':mobile_half_aux,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV2_9':mobile_half_9,
    'MobileNetV3':mobilev3_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'mobile_vit_xx_small_init': mobile_vit_xx_small_init,
    'mobile_vit_x_small_init': mobile_vit_x_small_init,
    'mobile_vit_small_init': mobile_vit_small_init,
    'mobile_vit_tiny_init': mobile_vit_tiny_init,
    'mobile_vit_tiny': mobile_vit_tiny,
    'mobile_vit_xx_small_in7': mobile_vit_xx_small_in7,
    'mobile_vit_x_small_in7': mobile_vit_x_small_in7,
    'mobile_vit_small_in7': mobile_vit_small_in7,
    'mobile_vit_tiny_in7': mobile_vit_tiny_in7,
    'mobile_vit_tiny_likevit': mobile_vit_tiny_likevit,
    'mobile_vit_tiny_novit':mobile_vit_tiny_novit,
    'mobile_vit_xx_small_best':mobile_vit_xx_small_best,
    'mobile_vit_tiny_best': mobile_vit_tiny_best,
    'mobile_vit_tiny_likevitpp':mobile_vit_tiny_likevitpp,
    'mobile_vit_tiny_novit_test1':mobile_vit_tiny_novit_test1,
    'mobile_vit_tiny_novit_test2':mobile_vit_tiny_novit_test2,
    'mobile_vit_tiny_novit_test3':mobile_vit_tiny_novit_test3,
    'mobile_vit_tiny_novit_test4':mobile_vit_tiny_novit_test4,
    'mobile_vit_tiny_novit_test5':mobile_vit_tiny_novit_test5,
    'mobile_vit_tiny_novit_test6':mobile_vit_tiny_novit_test6,
    'mobile_vit_tiny_novit_test0':mobile_vit_tiny_novit_test0,
    'mobile_vit_tiny_dilatedblock_5':mobile_vit_tiny_dilatedblock_5,
    'mobile_vit_tiny_dilatedblock_5_1':mobile_vit_tiny_dilatedblock_5_1,
    'mobile_vit_tiny_dilatedblock_5_2':mobile_vit_tiny_dilatedblock_5_2,
    'mobile_vit_tiny_dilatedblock_5_3':mobile_vit_tiny_dilatedblock_5_3,
    'mobile_vit_tiny_dilatedblock_5_4':mobile_vit_tiny_dilatedblock_5_4,
    'mobile_vit_tiny_dilatedblock_5_5':mobile_vit_tiny_dilatedblock_5_5,
    'mobile_vit_tiny_dilatedblock_5_5_1':mobile_vit_tiny_dilatedblock_5_5_1,
    'mobile_vit_tiny_dilatedblock_7':mobile_vit_tiny_dilatedblock_7,
    'mobile_vit_tiny_dilatedblock_9':mobile_vit_tiny_dilatedblock_9,
    'mobile_vit_tiny_dilatedblock_5_noweight':mobile_vit_tiny_dilatedblock_5_noweight,
    'SwiftFormer_XXS':SwiftFormer_XXS,
    'edgenext_xxx_small':edgenext_xxx_small,
    'mobilecontainer':mobilecontainer,
    'repvit_m0_6':repvit_m0_6,
    'repvit_m0_6_infiuni':repvit_m0_6_infiuni,
    'RepInfiniteUniVit_initial':repvit_m0_6_uni_initial,
    'repvit_m0_6_uni_ours':repvit_m0_6_uni_ours,
    'mobile_half_1_1_1':mobile_half_1_1_1,
    'mobile_half_1_1_1_1':mobile_half_1_1_1_1,
    'mobile_half_1_1_1_2':mobile_half_1_1_1_2,
    'mobile_half_1_1_2':mobile_half_1_1_2,
    'mobile_half_1_1_3':mobile_half_1_1_3,
    'mobile_half_1_1_4':mobile_half_1_1_4,
    'mobile_half_1_1_5':mobile_half_1_1_5,
    'mobile_half_1_1_7':mobile_half_1_1_7,
    'mobile_half_1_1_8':mobile_half_1_1_8,
    'mobile_half_1_1_8_1':mobile_half_1_1_8_1,
    'mobile_half_1_1_8_2':mobile_half_1_1_8_2,
    'mobile_half_1_1_8_3':mobile_half_1_1_8_3,
    'mobile_half_1_1_8_4':mobile_half_1_1_8_4,
    'mobile_half_1_1_8_5':mobile_half_1_1_8_5,
    'mobile_half_1_1_8_6':mobile_half_1_1_8_6,
    'mobile_half_1_1_8_7':mobile_half_1_1_8_7,
    'mobile_half_1_1_9':mobile_half_1_1_9,
    'mobile_half_1_1_10':mobile_half_1_1_10,
    'mobile_half_1_1_11':mobile_half_1_1_11,
    'mobile_half_1_1_12':mobile_half_1_1_12,
    'mobile_half_1_2_1':mobile_half_1_2_1,
    'mobile_half_1_2_1_1':mobile_half_1_2_1_1,
    'mobile_half_1_2_2':mobile_half_1_2_2,
    'mobile_half_1_2_3':mobile_half_1_2_3,
    'mobile_half_1_2_4':mobile_half_1_2_4,
    'mobile_half_1_3_1':mobile_half_1_3_1,
    'mobile_half_2_1_1':mobile_half_2_1_1,
    'mobile_half_2_1_1_1':mobile_half_2_1_1_1,
    'mobile_half_2_1_3':mobile_half_2_1_3,
    'mobile_half_2_1_4':mobile_half_2_1_4,
    'mobile_half_2_1_2':mobile_half_2_1_2,
    'mobile_half_2_1_5':mobile_half_2_1_5,
    'mobile_half_2_1_6':mobile_half_2_1_6,
    'mobile_half_3_1_1':mobile_half_3_1_1,
    'mobile_half_3_1_2':mobile_half_3_1_2,
    'mobile_half_4_1_1':mobile_half_4_1_1,
    'mobile_half_4_1_2':mobile_half_4_1_2,
    'mobile_half_4_1_2_1':mobile_half_4_1_2_1,
    'mobile_half_4_1_2_2':mobile_half_4_1_2_2,
    'mobile_half_4_1_3':mobile_half_4_1_3,
    'mobile_half_4_1_3_1':mobile_half_4_1_3_1,
    'mobile_half_4_1_3_2':mobile_half_4_1_3_2,
    'mobile_half_4_1_3_3':mobile_half_4_1_3_3,
    'mobile_half_4_1_4':mobile_half_4_1_4,
    'mobile_half_4_1_5':mobile_half_4_1_5,
    'mobile_half_4_1_6':mobile_half_4_1_6,
    'mobile_half_4_1_7':mobile_half_4_1_7,
    'mobile_half_5_1_1':mobile_half_5_1_1,
    'mobile_half_5_1_1_1':mobile_half_5_1_1_1,
    'mobile_half_5_1_1_2':mobile_half_5_1_1_2,
    'mobile_half_5_1_1_3':mobile_half_5_1_1_3,
    'mobile_half_5_1_1_4':mobile_half_5_1_1_4,
    'mobile_half_percent':mobile_half_percent,
    'mobile_half_6_1_1_1':mobile_half_6_1_1_1,
    'mobile_half_6_1_1_2':mobile_half_6_1_1_2,
    'mobile_half_6_1_1_2_1':mobile_half_6_1_1_2_1,
    'mobile_half_6_1_2_1':mobile_half_6_1_2_1,
    'mobile_half_1_1_12_1':mobile_half_1_1_12_1,
    'mobile_half_1_1_5_1':mobile_half_1_1_5_1,
    'mobile_half_1_1_5_2':mobile_half_1_1_5_2,
    'mobile_half_1_1_5_3':mobile_half_1_1_5_3,
    'mobile_half_1_1_5_3_1':mobile_half_1_1_5_3_1,
    'mobile_half_1_1_5_4':mobile_half_1_1_5_4,
    'mobile_half_1_2_2_1':mobile_half_1_2_2_1,
    'mobile_half_7_1_1':mobile_half_7_1_1,
    'mobile_half_7_1_2':mobile_half_7_1_2,
    'mobile_half_7_1_1_1':mobile_half_7_1_1_1,
    'mobile_half_7_1_2_1':mobile_half_7_1_2_1,
    'mobilemetanet':mobilemetanet,
    'mobilemetanet_1':mobilemetanet_1,
    'mobilemetanet_2':mobilemetanet_2,
    'RepTinynet':RepTinynet,
    'RepTinynet1':RepTinynet1,
    'RepTinynet2':RepTinynet2,
    'RepTinynet3':RepTinynet3,
    'RepTinynet4':RepTinynet4,
    'RepTinynet5':RepTinynet5,
    'RepTinynet6':RepTinynet6,
    'RepTinynet7':RepTinynet7,
    'RepTinynet8':RepTinynet8,
    'RepTinynet9':RepTinynet9,
    'RepTinynet10':RepTinynet10,
    'RepTinynet11':RepTinynet11,
    'RepTinynet12':RepTinynet12,
    'RepTinynet13':RepTinynet13,
    'RepTinynet14':RepTinynet14,
    'RepTinynet15':RepTinynet15,
    'RepTinynet16':RepTinynet16,
    'RepTinynet17':RepTinynet17,
    'RepTinynet18':RepTinynet18,
    'RepTinynet19':RepTinynet19,
    'RepTinynet20':RepTinynet20,
    'RepTinynet21':RepTinynet21,
    'RepTinynet22':RepTinynet22,
    'RepTinynet23':RepTinynet23,
    'RepTinynet24':RepTinynet24,
    'RepTinynet25':RepTinynet25,
    'mcunetlike':mcunetlike,
    'mobile_half_class':mobile_half_class,
    'mobile_half_class_2':mobile_half_class_2,
    'mobile_half_class_3':mobile_half_class_3,
    'repvggnet':repvggnet,
    'McuNetv1':MCUnetv1,
    'ContrastNet1':ContrastNet1,
    'ContrastNet2':ContrastNet2,
    'ContrastNet3':ContrastNet3,
    'ContrastNet4':ContrastNet4,
    'ContrastNet5':ContrastNet5,
    'ContrastNet6':ContrastNet6,
    'ContrastNet7':ContrastNet7,
    'ContrastNet8':ContrastNet8,
    'ContrastNet9':ContrastNet9,
    'ContrastNet10':ContrastNet10,
    'ContrastNet11':ContrastNet11,
    'ContrastNet12':ContrastNet12,
    'finalNet':finalNet,
    'squeezenet1':squeezenet1,



}
