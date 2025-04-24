# python train.py --model=mobile_vit_small
# python train.py --model resnet32 --arch_name resnet32
# python train.py --model mobile_vit_small_init --weights ./pre/mobilevit_s.pt  --arch_name mobile_vit_small_init
# python train.py --model mobile_vit_xx_small_init --weights ./pre/mobilevit_xxs.pt --arch_name mobile_vit_xx_small_init
# python train.py --model mobile_vit_tiny_init  --arch_name mobile_vit_tiny_init

# python train.py --model ShuffleV2 --arch_name ShuffleV2
# python train.py --model ShuffleV1 --arch_name ShuffleV1
# python train.py --model MobileNetV2 --arch_name MobileNetV2

# python train.py --model mobile_vit_tiny_in7 --arch_name mobile_vit_tiny_in7
# python train.py --model mobile_vit_tiny_novit --arch_name mobile_vit_tiny_novit

# 能否和/root/distill/log/mobile_vit_tiny_likevitlog_rank0.txt类似。因为这里主要就是想代替transformer模块。后两个就是看一下固定那3个最优点加rep后的效果
# python train.py --model mobile_vit_tiny_novit --arch_name mobile_vit_tiny_novit_5_3
# python train.py --model mobile_vit_xx_small_best --arch_name mobile_vit_xx_small_best --weights ./pre/mobilevit_xxs.pt
# python train.py --model mobile_vit_tiny_best --arch_name mobile_vit_tiny_best
# python train.py --model mobile_vit_tiny_likevitpp --arch_name mobile_vit_tiny_likevitpp
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit --arch_name kd_mobile_vit_tiny_novit
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_best --arch_name kd_mobile_vit_tiny_best

# python train.py --model mobile_vit_tiny_novit_test0  --arch_name mobile_vit_tiny_novit_test0
# python train.py --model mobile_vit_tiny_novit_test1  --arch_name mobile_vit_tiny_novit_test1
# python train.py --model mobile_vit_tiny_novit_test2  --arch_name mobile_vit_tiny_novit_test2
# python train.py --model mobile_vit_tiny_novit_test3  --arch_name mobile_vit_tiny_novit_test3
# python train.py --model mobile_vit_tiny_novit_test4  --arch_name mobile_vit_tiny_novit_test4
# python train.py --model mobile_vit_tiny_novit_test5  --arch_name mobile_vit_tiny_novit_test5
# python train.py --model mobile_vit_tiny_novit_test6  --arch_name mobile_vit_tiny_novit_test6
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test0 --arch_name mobile_vit_tiny_novit_test0
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test1 --arch_name mobile_vit_tiny_novit_test1
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test2 --arch_name mobile_vit_tiny_novit_test2
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test3 --arch_name kd_mobile_vit_tiny_novit_test3
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test4 --arch_name kd_mobile_vit_tiny_novit_test4
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test5 --arch_name kd_mobile_vit_tiny_novit_test5
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit_test6 --arch_name kd_mobile_vit_tiny_novit_test6

# python train.py --model mobile_vit_tiny_dilatedblock_5_1  --arch_name mobile_vit_tiny_dilatedblock_5_1
# python train.py --model mobile_vit_tiny_dilatedblock_5_2  --arch_name mobile_vit_tiny_dilatedblock_5_2
# python train.py --model mobile_vit_tiny_dilatedblock_5_3  --arch_name mobile_vit_tiny_dilatedblock_5_3
# python train.py --model mobile_vit_tiny_dilatedblock_5_4  --arch_name mobile_vit_tiny_dilatedblock_5_4
# python train.py --model mobile_vit_tiny_dilatedblock_5_5  --arch_name mobile_vit_tiny_dilatedblock_5_5
python train.py --model mobile_vit_tiny_dilatedblock_5_noweight  --arch_name mobile_vit_tiny_dilatedblock_5_noweight


# python train.py --model ShuffleV2  --arch_name shufflenetv2_tiny
# python train.py --model ShuffleV1  --arch_name shufflenetv1_tiny_mini  --trial 1

python train.py --model ShuffleV2  --arch_name shufflenetv2_tiny_mini  --trial 2
python train.py --model mobile_vit_tiny  --arch_name mobile_vit_tiny_mini --trial 1
# 后2改mv2
python train.py --model mobile_vit_tiny_likevit  --arch_name mobile_vit_tiny_likevit_mini --trial 1
# 看batch——size的影响
python train.py --model mobile_vit_tiny_likevit  --arch_name mobile_vit_tiny_likevit_mini2 --trial 2 --batch_size 64
# mobile_vit_tiny_novit 无自注意力机制
python train.py --model mobile_vit_tiny_novit  --arch_name mobile_vit_tiny_novit_mini --trial 1
# mobile_half
python train.py --model MobileNetV2  --arch_name MobileNetV2_mini --trial 1
# mobile_vit_tiny_dilatedblock_5_3
python train.py --model mobile_vit_tiny_dilatedblock_5_3  --arch_name mobile_vit_tiny_dilatedblock_5_3_mini --trial 1 --batch_size 64
# mcu
python train.py --model mcunet  --arch_name mcunet_mini2 --trial 2
# SwiftFormer_XXS
python train.py --model SwiftFormer_XXS  --arch_name SwiftFormer_XXS_mini2 --trial 1
# edgenext_xxx_small
python train.py --model edgenext_xxx_small  --arch_name edgenext_xxx_small_mini --trial 1
# mobile_vit_tiny_best
python train.py --model mobile_vit_tiny_best  --arch_name mobile_vit_tiny_best_mini --trial 2
# /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_init --arch_name kd_mobile_vit_tiny_init_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
mobile_vit_tiny
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny --arch_name kd_mobile_vit_tiny_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
# mobile_vit_tiny_likevit
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_likevit --arch_name kd_mobile_vit_tiny_likevit_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
ShuffleV1
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s ShuffleV1 --arch_name kd_ShuffleV1_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
ShuffleV2
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s ShuffleV2 --arch_name kd_ShuffleV2_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
mobile_vit_tiny_dilatedblock_5_3
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_3 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_mini13 --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_3 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_mini5 --trial 2 --batch_size 64 -r 0.1 -a 0.9 -b 0  --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny --arch_name kd_mobile_vit_tiny_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
# mobile_half
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s MobileNetV2 --arch_name kd_MobileNetV2_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0  --dataset miniImage
mobile_vit_tiny_best
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_best --arch_name kd_mobile_vit_tiny_best_mini --trial 2 --batch_size 64 -r 0.1 -a 0.9 -b 0  --dataset miniImage
mobile_vit_tiny_dilatedblock_5_4
# python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_4 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_mini13 --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_4 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_mini13 --trial 2 --batch_size 64 -r 0.1 -a 0.9 -b 0  --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny --arch_name kd_mobile_vit_tiny_mini --trial 1 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset miniImage
python train.py --model mobile_vit_tiny_dilatedblock_5_4  --arch_name mobile_vit_tiny_dilatedblock_5_4_mini --trial 2

mobilecontainer
python train.py --model mobilecontainer  --arch_name mobilecontainer --trial 2

# 由于没有下采样，所以需要重新做实验。为了缩短实验时间，将epoch先缩短至80，如果实验结果为预期，再扩大至240。直接做kd先看最终结果吧。
实验对象是mobile_vit_tiny_init   mobile_vit_tiny_dilatedblock_5_5 mobile_vit_tiny_dilatedblock_5_3
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_init --arch_name kd_mobile_vit_tiny_init_mini100 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 3 --dataset miniImage --epochs 80
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_5 --arch_name kd_mobile_vit_tiny_dilatedblock_5_5_mini100 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 3 --dataset miniImage --epochs 80
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_3 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_mini100 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 3 --dataset miniImage --epochs 80
image size不下14
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_5 --arch_name kd_mobile_vit_tiny_dilatedblock_5_5_mini100_14 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 4 --dataset miniImage --epochs 80
down kernel 变成13
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_5 --arch_name kd_mobile_vit_tiny_dilatedblock_5_5_mini100_14_13 --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 4 --dataset miniImage --epochs 80
# mobile_vit_tiny_dilatedblock_5_3 -cifar 4 把layer 3 4的block 变为了8 3
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_3 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_modified83 -r 0.1 -a 0.9 -b 0 --trial 4 --dataset cifar100
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_3 --arch_name kd_mobile_vit_tiny_dilatedblock_5_3_modified83nolarge -r 0.1 -a 0.9 -b 0 --trial 5 --dataset cifar100
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_init --arch_name kd_mobile_vit_tiny_init_modifiedreduce1 -r 0.1 -a 0.9 -b 0 --trial 5 --dataset cifar100
layer 3 没有large kernel
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_2 --arch_name kd_mobile_vit_tiny_dilatedblock_5_2_modifiednolarge -r 0.1 -a 0.9 -b 0 --trial 4 --dataset cifar100
layer 3变48 channel
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_2 --arch_name kd_mobile_vit_tiny_dilatedblock_5_2_modifiedmorechannel -r 0.1 -a 0.9 -b 0 --trial 4 --dataset cifar100
最初的错误设定
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_2 --arch_name kd_mobile_vit_tiny_dilatedblock_5_2_modifiedinilnolarge -r 0.1 -a 0.9 -b 0 --trial 4 --dataset cifar100
mobile_vit_tiny layer 3的stride改为1
kd_mobile_vit_tiny_dilatedblock_5_5_1_mini100_5formerlog_rank0
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_dilatedblock_5_5_1 --arch_name kd_mobile_vit_tiny_dilatedblock_5_5_1_mini100_5former --batch_size 64 -r 0.1 -a 0.9 -b 0 --trial 4 --dataset miniImage --epochs 80
cifar100的下面三个，将layer3的stride都变1。分别是mobile_vit_tiny_init，mobile_vit_tiny和mobile_vit_tiny_likevit以及mobile_vit_tiny_novit，mobile_vit_tiny_best,MobileNetV2
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_init --arch_name kd_mobile_vit_tiny_init_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100
62.9800/67.1600---res4 69.2699
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny --arch_name kd_mobile_vit_tiny_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100
63.4800/67.4100---res4 69.0699
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_likevit --arch_name kd_mobile_vit_tiny_likevit_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100
65.8000/67.8700---res4 70.5 
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_novit --arch_name kd_mobile_vit_tiny_novit_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100
62.1400/69.5400---res4 67.3
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s mobile_vit_tiny_best --arch_name kd_mobile_vit_tiny_best_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100
64.5000/67.1500---res4 69.86
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s MobileNetV2 --arch_name kd_MobileNetV2_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100
61.8600/68.25---res4 71.4
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_cifar100_lr_0.01_decay_0.0005_trial_0/mobile_vit_small_init_best.pth --model_s MobileNetV3 --arch_name kd_MobileNetV3_layer31 -r 0.1 -a 0.9 -b 0 --trial 6 --dataset cifar100


python train.py  --model mobile_vit_tiny_init --arch_name mobile_vit_tiny_init_layer31  --trial 6 --dataset cifar100
67.1600
python train.py  --model mobile_vit_tiny --arch_name mobile_vit_tiny_layer31  --trial 6 --dataset cifar100
67.4100
python train.py  --model mobile_vit_tiny_likevit --arch_name mobile_vit_tiny_likevit_layer31  --trial 6 --dataset cifar100
67.8700
python train.py  --model mobile_vit_tiny_novit --arch_name mobile_vit_tiny_novit_layer31 --trial 6 --dataset cifar100
69.5400
python train.py --model mobile_vit_tiny_best --arch_name mobile_vit_tiny_best_layer31 --trial 6 --dataset cifar100
67.1500
python train.py  --model MobileNetV2 --arch_name MobileNetV2_layer31 --trial 6 --dataset cifar100
mobile_vit_small_init
python train.py  --model mobile_vit_xx_small_init --arch_name mobile_vit_xx_small_init --trial 6 --dataset cifar100
MobileNetV3
python train.py  --model MobileNetV3 --arch_name MobileNetV3_init --trial 6 --dataset cifar100
mobile_vit_small_init --方便更准确的dis
python train.py  --model mobile_vit_small_init --arch_name mobile_vit_small_init_new --trial 6 --dataset cifar100
repvit_m0_6
python train.py  --model repvit_m0_6 --arch_name repvit_m0_6_init --trial 6 --dataset cifar100
python train.py  --model ShuffleV1 --arch_name ShuffleV1 --trial 6 --dataset cifar100
69.52999877929688 
python train.py  --model ShuffleV2 --arch_name ShuffleV2 --trial 6 --dataset cifar100
66.48999786376953
MobileNetV2--resolution最终为4
python train.py  --model MobileNetV2 --arch_name MobileNetV2_4 --trial 6 --dataset cifar100
python train.py  --model MobileNetV2 --arch_name MobileNetV2_4_first2 --trial 6 --dataset cifar100
71.4000015258
最终resolution为4
python train.py  --model mobile_vit_small_init --arch_name mobile_vit_small_init_new --trial 6 --dataset cifar100
python train.py  --model mobile_vit_tiny_init --arch_name mobile_vit_tiny_init_res4 --trial 7 --dataset cifar100
python train.py  --model mobile_vit_tiny --arch_name mobile_vit_tiny_res4 --trial 7 --dataset cifar100
python train.py  --model mobile_vit_tiny_likevit --arch_name mobile_vit_tiny_likevit_res4 --trial 7 --dataset cifar100
python train.py  --model mobile_vit_tiny_novit --arch_name mobile_vit_tiny_novit_res4 --trial 7 --dataset cifar100
python train.py  --model mobile_vit_tiny_best --arch_name mobile_vit_tiny_best_res4 --trial 7 --dataset cifar100

imagenet--repvit_m0_6
python train.py  --model repvit_m0_6 --arch_name repvit_m0_6_init_mini_80 --trial 6 --dataset miniImage --batch_size 64 --epochs 80
repvit_m0_6_infiuni--0.9M--3size的dw换13
python train.py  --model repvit_m0_6_infiuni --arch_name repvit_m0_6_infiuni_mini_80 --trial 6 --dataset miniImage --batch_size 64 --epochs 80
RepInfiniteUniVit_initial--原始的uni模块
python train.py  --model RepInfiniteUniVit_initial --arch_name RepInfiniteUniVit_initial_mini_80 --trial 6 --dataset miniImage --batch_size 64 --epochs 80
python train.py  --model RepInfiniteUniVit_initial --arch_name RepInfiniteUniVit_initial_mini_80_mostuni --trial 7 --dataset miniImage --batch_size 64 --epochs 80
repvit_m0_6_uni_ours -- weight模块
python train.py  --model repvit_m0_6_uni_ours --arch_name repvit_m0_6_uni_ours_mini_80 --trial 6 --dataset miniImage --batch_size 64 --epochs 80

repvit_m0_6_init_mini_80  73.1167;RepInfiniteUniVit_initial_mini_80  70.40;repvit_m0_6_uni_ours_mini_80 71.3583;RepInfiniteUniVit_initial_mini_80_mostuni  69.1333;repvit_m0_6_infiuni_mini_80  70.7083 冒然扩大kernel_size，一般不会有利于性能提升


mobilenetv2探究rep
1. stem
python train.py  --model mobile_half_1_1_1 --arch_name Mobile_half_1_1_1 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_1_2 --arch_name Mobile_half_1_1_2 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_1_3 --arch_name Mobile_half_1_1_3 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_1_4 --arch_name Mobile_half_1_1_4 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_1_5 --arch_name Mobile_half_1_1_5 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_1_5 --arch_name Mobile_half_1_1_6 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_2_1 --arch_name Mobile_half_1_2_1 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_2_2 --arch_name Mobile_half_1_2_2 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_2_3 --arch_name Mobile_half_1_2_3 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_2_4 --arch_name Mobile_half_1_2_4 --trial 7 --dataset cifar100
python train.py  --model mobile_half_1_3_1 --arch_name Mobile_half_1_3_1 --trial 7 --dataset cifar100

2.down
python train.py  --model mobile_half_2_1_1 --arch_name Mobile_half_2_1_1 --trial 7 --dataset cifar100
python train.py  --model mobile_half_2_1_1_1 --arch_name Mobile_half_2_1_1_1 --trial 7 --dataset cifar100
python train.py  --model mobile_half_2_1_2 --arch_name Mobile_half_2_1_2 --trial 7 --dataset cifar100
python train.py  --model mobile_half_2_1_3 --arch_name Mobile_half_2_1_3 --trial 7 --dataset cifar100
python train.py  --model mobile_half_2_1_4 --arch_name Mobile_half_2_1_4 --trial 7 --dataset cifar100
python train.py  --model mobile_half_2_1_5 --arch_name Mobile_half_2_1_5 --trial 7 --dataset cifar100
python train.py  --model mobile_half_2_1_6 --arch_name Mobile_half_2_1_6_1 --trial 7 --dataset cifar100

3 class
mobile_half_3_1_1
python train.py  --model mobile_half_3_1_1 --arch_name Mobile_half_3_1_1 --trial 7 --dataset cifar100

4 main
python train.py  --model mobile_half_4_1_1 --arch_name Mobile_half_4_1_1 --trial 7 --dataset cifar100
python train.py  --model mobile_half_4_1_2 --arch_name Mobile_half_4_1_2 --trial 7 --dataset cifar100
python train.py  --model mobile_half_4_1_3 --arch_name Mobile_half_4_1_3 --trial 7 --dataset cifar100
python train.py  --model mobile_half_4_1_4 --arch_name Mobile_half_4_1_4 --trial 7 --dataset cifar100
python train.py  --model mobile_half_4_1_5 --arch_name Mobile_half_4_1_5 --trial 7 --dataset cifar100
python train.py  --model mobile_half_4_1_6 --arch_name Mobile_half_4_1_6 --trial 7 --dataset cifar100
python train.py  --model mobile_half_4_1_7 --arch_name Mobile_half_4_1_7 --trial 7 --dataset cifar100

5.all 
mobile_half_5_1_1
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1 --trial 7 --dataset cifar100
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1_mini --trial 8 --dataset miniImage --batch_size 64 --epochs 80 --cuda 0
--trial 8 --dataset miniImage --batch_size 64 --epochs 80 --cuda 0
mini数据集的


percent
mobile_half_percent
python train.py  --model mobile_half_percent --arch_name mobile_half_percent --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_percent --arch_name mobile_half_percent_se --trial 2 --dataset cifar100 --cuda 1


2024 3-18
python train.py  --model mobile_half_1_1_2 --arch_name Mobile_half_1_1_2_mini_240_2 --trial 2 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_7 --arch_name Mobile_half_1_1_7_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_8 --arch_name Mobile_half_1_1_8_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_9 --arch_name Mobile_half_1_1_9_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_10 --arch_name Mobile_half_1_1_10_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_11 --arch_name Mobile_half_1_1_11_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_12 --arch_name Mobile_half_1_1_12_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_1_1 --arch_name Mobile_half_1_1_1_1_mini_240 --trial 1 --dataset miniImage --cuda 1

有weight
python train.py  --model mobile_half_1_1_1 --arch_name Mobile_half_1_1_1_mini_240_1 --trial 0 --dataset miniImage --cuda 1
有weight
python train.py  --model mobile_half_1_1_12 --arch_name Mobile_half_1_1_12_mini_240_1 --trial 0 --dataset miniImage --cuda 1
mobile_half_4_1_2_1
python train.py  --model mobile_half_4_1_2_1 --arch_name Mobile_half_4_1_2_1_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_4_1_2_2 --arch_name Mobile_half_4_1_2_2_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_4_1_4 --arch_name Mobile_half_4_1_4_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_4_1_5 --arch_name Mobile_half_4_1_5_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_4_1_3_1 --arch_name Mobile_half_4_1_3_1_mini_240 --trial 0 --dataset miniImage --cuda 1
没算力
python train.py  --model mobile_half_4_1_3_3 --arch_name Mobile_half_4_1_3_3_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_4_1_3_2 --arch_name Mobile_half_4_1_3_2_mini_240 --trial 0 --dataset miniImage --cuda 1

3-20
python train.py  --model MobileNetV2 --arch_name MobileNetV2_mini_240_seed --trial 24 --dataset miniImage --cuda 0

stem在mini中的最佳结果。
python train.py  --model mobile_half_1_1_8_1 --arch_name Mobile_half_1_1_8_1_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_8_2 --arch_name Mobile_half_1_1_8_2_mini_240 --trial 0 --dataset miniImage --cuda 1
mobile_half_1_1_12_1--新的weights动态赋予方式
python train.py  --model mobile_half_1_1_12_1 --arch_name Mobile_half_1_1_12_1_mini_240 --trial 0 --dataset miniImage --cuda 1

，其次减expand加width直接减少冗余。
python train.py  --model mobile_half_6_1_2_1 --arch_name Mobile_half_6_1_2_1_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1_240_seed --trial 9 --dataset miniImage --cuda 0

测试通道数的影响。先试用group-rep来增加维度的独立性特征信息
python train.py  --model mobile_half_6_1_1_1 --arch_name Mobile_half_6_1_1_1_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_6_1_1_2 --arch_name Mobile_half_6_1_1_2_mini_240 --trial 0 --dataset miniImage --cuda 1
mobile_half_6_1_1_2_1 weight 均分
python train.py  --model mobile_half_6_1_1_2_1 --arch_name Mobile_half_6_1_1_2_1_mini_240 --trial 0 --dataset miniImage --cuda 1
mobile_half_1_1_8_3--把uni后面提取各个感受野的1x1去掉
python train.py  --model mobile_half_1_1_8_3 --arch_name Mobile_half_1_1_8_3_mini_240 --trial 0 --dataset miniImage --cuda 0

cifar
python train.py  --model mobile_half_6_1_1_2_1 --arch_name Mobile_half_6_1_1_2_1_cifar --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_6_1_1_2 --arch_name Mobile_half_6_1_1_2_cifar --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_6_1_1_1 --arch_name Mobile_half_6_1_1_1_cifar --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_6_1_1_1 --arch_name Mobile_half_6_1_2_1_cifar --trial 1 --dataset cifar100 --cuda 0  4-0.6
python train.py  --model mobile_half_6_1_1_1 --arch_name Mobile_half_6_1_2_1_cifar_1 --trial 2 --dataset cifar100 --cuda 1  2-0.75
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1_cifar_seed --trial 1 --dataset cifar100 --cuda 0
python train.py  --model MobileNetV2 --arch_name MobileNetV2_cifar_seed --trial 1 --dataset cifar100 --cuda 0

3-23
python train.py  --model mobile_half_1_1_3 --arch_name Mobile_half_1_1_3_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_1 --arch_name Mobile_half_1_1_1_mini_240_1 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_5 --arch_name Mobile_half_1_1_5_mini_240 --trial 1 --dataset miniImage --cuda 1
'mobile_half_1_1_5_1','mobile_half_1_1_5_2','mobile_half_1_1_5_3','mobile_half_1_1_5_4','mobile_half_1_1_5_3_1'
python train.py  --model mobile_half_1_1_5_1 --arch_name Mobile_half_1_1_5_1_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_5_2 --arch_name Mobile_half_1_1_5_2_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_5_3 --arch_name Mobile_half_1_1_5_3_mini_240 --trial 1 --dataset miniImage --cuda 0

python train.py  --model mobile_half_1_1_5_3_1 --arch_name Mobile_half_1_1_5_3_1_mini_240 --trial 1 --dataset miniImage --cuda 1

python train.py  --model mobile_half_1_1_5_4 --arch_name Mobile_half_1_1_5_4_mini_240 --trial 1 --dataset miniImage --cuda 1
mobile_half_1_2_2_1
python train.py  --model mobile_half_1_2_2_1 --arch_name Mobile_half_1_2_2_1_mini_240 --trial 1 --dataset miniImage --cuda 1
mobile_half_1_2_1_1
python train.py  --model mobile_half_1_2_1_1 --arch_name Mobile_half_1_2_1_1_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_2_1_6 --arch_name Mobile_half_2_1_6_seed_cifar --trial 7 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_2_1_6 --arch_name Mobile_half_2_1_6_seed --trial 1 --dataset miniImage --cuda 0


3-25
13的感受野，但是代码写得是11，重跑
python train.py  --model mobile_half_1_1_5_3 --arch_name Mobile_half_1_1_5_3_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_5_3_1 --arch_name Mobile_half_1_1_5_3_1_mini_240 --trial 1 --dataset miniImage --cuda 0
mobile_half_7_1_1
python train.py  --model mobile_half_7_1_1 --arch_name Mobile_half_7_1_1_cifar --trial 2 --dataset cifar100 --cuda 1
记得该模型再跑
python train.py  --model mobile_half_7_1_1 --arch_name Mobile_half_7_1_1_mini_240 --trial 1 --dataset miniImage --cuda 1

python train.py  --model mobile_half_7_1_2 --arch_name Mobile_half_7_1_2_cifar --trial 2 --dataset cifar100 --cuda 1
记得该模型再跑
python train.py  --model mobile_half_7_1_2 --arch_name Mobile_half_7_1_2_mini_240 --trial 1 --dataset miniImage --cuda 1

python train.py  --model mobile_half_7_1_1_1 --arch_name Mobile_half_7_1_1_1_cifar --trial 2 --dataset cifar100 --cuda 0
记得该模型再跑
python train.py  --model mobile_half_7_1_1_1 --arch_name Mobile_half_7_1_1_1_mini_240 --trial 1 --dataset miniImage --cuda 0

python train.py  --model mobile_half_7_1_2_1 --arch_name Mobile_half_7_1_2_1_cifar --trial 2 --dataset cifar100 --cuda 1
记得该模型再跑
python train.py  --model mobile_half_7_1_2_1 --arch_name Mobile_half_7_1_2_1_mini_240 --trial 1 --dataset miniImage --cuda 1


3-26
python train.py  --model mobile_half_4_1_2_1 --arch_name Mobile_half_4_1_2_1_cifar_240 --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_4_1_2_2 --arch_name Mobile_half_4_1_2_2_cifar_240 --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_4_1_3_1 --arch_name Mobile_half_4_1_3_1_cifar_240 --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_4_1_3_3 --arch_name Mobile_half_4_1_3_3_cifar_240 --trial 1 --dataset cifar100 --cuda 0
python train.py  --model mobile_half_4_1_3_2 --arch_name Mobile_half_4_1_3_2_cifar_240 --trial 1 --dataset cifar100 --cuda 0


9---修改classifier
python train_9.py  --model MobileNetV2_9 --arch_name MobileNetV2_999_cifar_seed --trial 9 --dataset cifar100 --cuda 0

mobile_half_1_1_1_2  3-29
python train.py  --model mobile_half_1_1_1_2 --arch_name Mobile_half_1_1_1_2_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_1_2 --arch_name Mobile_half_1_1_1_2_cifar_240 --trial 2 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_5_1_1_1 --arch_name Mobile_half_5_1_1_1_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_5_1_1_1 --arch_name Mobile_half_5_1_1_1_cifar_240 --trial 2 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_5_1_1_2 --arch_name Mobile_half_5_1_1_2_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_5_1_1_2 --arch_name Mobile_half_5_1_1_2_cifar_240 --trial 2 --dataset cifar100 --cuda 1

4-01
python train.py  --model mobile_half_1_2_2_1 --arch_name Mobile_half_1_2_2_1_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_7 --arch_name Mobile_half_1_1_7_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_8 --arch_name Mobile_half_1_1_8_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_8_1 --arch_name Mobile_half_1_1_8_1_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_8_2 --arch_name Mobile_half_1_1_8_2_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_8_3 --arch_name Mobile_half_1_1_8_3_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_9 --arch_name Mobile_half_1_1_9_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_10 --arch_name Mobile_half_1_1_10_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_11 --arch_name Mobile_half_1_1_11_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_12 --arch_name Mobile_half_1_1_12_cifar_240 --trial 1 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_1_1_12_1 --arch_name Mobile_half_1_1_12_1_cifar_240 --trial 1 --dataset cifar100 --cuda 1
# 还没写
python train.py  --model mobile_half_1_1_12_2 --arch_name Mobile_half_1_1_12_2_cifar_240 --trial 1 --dataset cifar100 --cuda 1

python train.py  --model mobile_half_1_1_8_4 --arch_name Mobile_half_1_1_8_4_mini_240bn --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_8 --arch_name Mobile_half_1_1_8_mini_240bn --trial 0 --dataset miniImage --cuda 1

python train.py  --model mobile_half_1_1_8_5 --arch_name Mobile_half_1_1_8_5_mini_240bn --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_8_6 --arch_name Mobile_half_1_1_8_6_mini_240bn --trial 0 --dataset miniImage --cuda 1

python train.py  --model mobile_half_1_1_8_7 --arch_name Mobile_half_1_1_8_7_mini_240bn --trial 0 --dataset miniImage --cuda 1


python train.py  --model mobile_half_1_1_8_1 --arch_name Mobile_half_1_1_8_1_mini_240bn --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_8_2 --arch_name Mobile_half_1_1_8_2_mini_240bn --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_3_1_2 --arch_name Mobile_half_3_1_2_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_3_1_2 --arch_name Mobile_half_3_1_2_cifar_240 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model mobilemetanet --arch_name Mobilemetanet_mini_240 --trial 0 --dataset miniImage --cuda 1
# tokenmixer没有跳跃连接
python train.py  --model mobilemetanet --arch_name Mobilemetanet_mini_240_1 --trial 0 --dataset miniImage --cuda 1
# # token mixer在上一个的基础上换rep
# python train.py  --model mobilemetanet --arch_name Mobilemetanet_mini_240_2 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobilemetanet_1 --arch_name Mobilemetanet_1_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobilemetanet_2 --arch_name Mobilemetanet_2_mini_240 --trial 0 --dataset miniImage --cuda 1

python train.py  --model mobilemetanet --arch_name Mobilemetanet_cifar_240 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model mobilemetanet_1 --arch_name Mobilemetanet_1_cifar_240 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model mobilemetanet_2 --arch_name Mobilemetanet_2_cifar_240 --trial 0 --dataset cifar100 --cuda 0

python train_9.py  --model MobileNetV2_9 --arch_name MobileNetV2_999_cifar_seed_reverse --trial 9 --dataset cifar100 --cuda 0
python train_9.py  --model MobileNetV2_9 --arch_name MobileNetV2_999_miniImage_seed_reverse --trial 9 --dataset miniImage --cuda 0

python train.py  --model mobile_half_1_1_8 --arch_name Mobile_half_1_1_8_mini_240seed --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_1_1_8 --arch_name Mobile_half_1_1_8_mini_240nobn --trial 0 --dataset miniImage --cuda 0

跑完meta，根据结果跑一下meta的后续以及1182有bn和没bn的seed版本。

python train.py  --model RepTinynet --arch_name RepTinynet_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet1 --arch_name RepTinynet1_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet2 --arch_name RepTinynet2_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet3 --arch_name RepTinynet3_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet4 --arch_name RepTinynet4_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet5 --arch_name RepTinynet5_mini_240 --trial 0 --dataset miniImage --cuda 1


4-04
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1_240_seed3_6 --trial 9 --dataset miniImage --cuda 0
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1_240_seed3_2 --trial 9 --dataset miniImage --cuda 0
python train.py  --model RepTinynet1 --arch_name RepTinynet1_mini_240_6 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet2 --arch_name RepTinynet2_mini_240_6 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet4 --arch_name RepTinynet4_mini_240_6 --trial 0 --dataset miniImage --cuda 1

4-05
python train.py  --model RepTinynet6 --arch_name RepTinynet4_mini_240_6 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet7 --arch_name RepTinynet7_mini_240_6 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet --arch_name RepTinynet_cifar_240_6 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model RepTinynet1 --arch_name RepTinynet1_cifar_240_6 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model RepTinynet2 --arch_name RepTinynet2_cifar_240_6 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model RepTinynet3 --arch_name RepTinynet3_cifar_240_6 --trial 0 --dataset cifar100 --cuda 0
python train.py  --model RepTinynet4 --arch_name RepTinynet4_cifar_240_6 --trial 0 --dataset cifar100 --cuda 1
python train.py  --model RepTinynet5 --arch_name RepTinynet5_cifar_240_6 --trial 0 --dataset cifar100 --cuda 1
python train.py  --model RepTinynet6 --arch_name RepTinynet6_cifar_240_6 --trial 0 --dataset cifar100 --cuda 1
python train.py  --model RepTinynet7 --arch_name RepTinynet7_cifar_240_6 --trial 0 --dataset cifar100 --cuda 1
python train.py  --model RepTinynet8 --arch_name RepTinynet8_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet9 --arch_name RepTinynet9_mini_240 --trial 0 --dataset miniImage --cuda 0

4-06
python train.py  --model mobile_half_1_1_5_3_1 --arch_name Mobile_half_1_1_5_3_1_mini_240_seed --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_1_3 --arch_name Mobile_half_1_1_3_mini_240_seed --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_1_2_2_1 --arch_name Mobile_half_1_2_2_1_mini_240_seed --trial 1 --dataset miniImage --cuda 1
python train.py  --model MobileNetV2 --arch_name MobileNetV2_mini_240_seed_278 --trial 24 --dataset miniImage --cuda 1
,'RepTinynet10','RepTinynet11','RepTinynet12','RepTinynet13'
python train.py  --model RepTinynet10 --arch_name RepTinynet10_mini_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet11 --arch_name RepTinynet11_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet12 --arch_name RepTinynet12_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet13 --arch_name RepTinynet13_mini_240 --trial 0 --dataset miniImage --cuda 0

python train.py  --model mcunetlike --arch_name mcunetlike_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_half_5_1_1 --arch_name Mobile_half_5_1_1_240_seed_se --trial 9 --dataset miniImage --cuda 0
python train.py  --model RepTinynet14 --arch_name RepTinynet14_mini_240_25 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet15 --arch_name RepTinynet15_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet16 --arch_name RepTinynet16_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet17 --arch_name RepTinynet17_mini_240 --trial 0 --dataset miniImage --cuda 1
'mobile_half_5_1_1_3
python train.py  --model mobile_half_5_1_1_3 --arch_name Mobile_half_5_1_1_3_mini_240 --trial 0 --dataset miniImage --cuda 0

4-08
python train.py  --model RepTinynet17 --arch_name RepTinynet17_cifar_240 --trial 0 --dataset cifar100 --cuda 1
python train.py  --model mobile_half_5_1_1_3 --arch_name Mobile_half_5_1_1_3_cifar_240 --trial 0 --dataset cifar100 --cuda 1
python train.py  --model mobilemetanet --arch_name Mobilemetanet_mini_240_262 --trial 0 --dataset miniImage --cuda 1

前面的实验conv2错误，重做：
python train.py  --model MobileNetV2 --arch_name MobileNetV2_mini_240_seed_605_class640 --trial 24 --dataset miniImage --cuda 1
python train.py  --model RepTinynet --arch_name RepTinynet_mini_240_class1280 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet9 --arch_name RepTinynet9_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet8 --arch_name RepTinynet8_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet10 --arch_name RepTinynet10_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet11 --arch_name RepTinynet11_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0

python train.py  --model RepTinynet12 --arch_name RepTinynet12_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet13 --arch_name RepTinynet13_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 1

---mingt pao
python train.py  --model RepTinynet14 --arch_name RepTinynet14_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet15 --arch_name RepTinynet15_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet16 --arch_name RepTinynet16_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet17 --arch_name RepTinynet17_mini_240_6_class1280 --trial 0 --dataset miniImage --cuda 0
以上还没跑，
4-09 先跑下面的额,'mobile_half_class','mobile_half_class_2','mobile_half_class_3',下面的代码是为了让classifier少占用模型比例。传统占模型参数比高达40%，如果分类层的输出受超参数控制，则可将比例下降为20%，如果采用下面的方法，比例可下降到2%左右
python train.py  --model mobile_half_class --arch_name Mobile_half_class_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_class --arch_name Mobile_half_class_240_481 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_class --arch_name Mobile_half_class_240_1114 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_class_2 --arch_name Mobile_half_class_2_240 --trial 0 --dataset miniImage --cuda 0
python train.py  --model mobile_half_class_3 --arch_name Mobile_half_class_3_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet18 --arch_name RepTinynet18_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet19 --arch_name RepTinynet19_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model MobileNetV2 --arch_name MobileNetV2_mini_240_seed_657 --trial 24 --dataset miniImage --cuda 1

python train.py  --model RepTinynet20 --arch_name RepTinynet20_mini_240 --trial 0 --dataset miniImage --cuda 1

python train.py  --model mobile_half_class --arch_name Mobile_half_class_240_1116 --trial 0 --dataset miniImage --cuda 0
重做，应该是先dropblock(input),再过卷积层。
python train.py  --model RepTinynet19 --arch_name RepTinynet19_mini_240_1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet19 --arch_name RepTinynet19_mini_240_2 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet14 --arch_name RepTinynet14_mini_240_change --trial 0 --dataset miniImage --cuda 0
还没跑
python train.py  --model RepTinynet14 --arch_name RepTinynet14_mini_240_change_2 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet21 --arch_name RepTinynet21_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet22 --arch_name RepTinynet22_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240 --trial 0 --dataset miniImage --cuda 1
mobile_half_5_1_1_4
python train.py  --model mobile_half_5_1_1_4 --arch_name Mobile_half_5_1_1_4_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet21 --arch_name RepTinynet21_mini_240_no1 --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet21 --arch_name RepTinynet21_mini_240_5 --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet21 --arch_name RepTinynet21_mini_240_9 --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_no_identity --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_no_1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_no_1x1_double_dense --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_double_dense --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_weight_155 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_only_double_dense --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_only_dw_155 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet22 --arch_name RepTinynet22_mini_240_onlydw --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet22 --arch_name RepTinynet22_mini_240_onlypw --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_only_triple_dense --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_only_fourth_dense --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_only_fifth_dense --trial 0 --dataset miniImage --cuda 0

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_only_double_dense_weight --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_usebn --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_no1x1 --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_noidentity --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_weight --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_doubledense --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_usebn_odoubledense --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_weight0_111 --trial 0 --dataset miniImage --cuda 1

4-15
python train.py  --model RepTinynet20 --arch_name RepTinynet20_mini_240_pre --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet20 --arch_name RepTinynet20_mini_240_pos --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet21 --arch_name RepTinynet21_mini_240_pos --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet21 --arch_name RepTinynet21_mini_240_pre --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_pw_dw_onlydense --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_pw_dw_onlydense_block3 --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_pw_dw_onlydense_block5 --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_pw_dw_onlydense_dif_init --trial 1 --dataset miniImage --cuda 0
meipao
python train.py  --model RepTinynet24 --arch_name RepTinynet24_mini_240_pw_dw_onlydense_weight --trial 1 --dataset miniImage --cuda 0

python train.py  --model mobile_half_class_3 --arch_name Mobile_half_class_3_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240 --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_relu6 --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_leakrelu --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_leakrelu01 --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_elu --trial 1 --dataset miniImage --cuda 1

python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_selu --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_gelu --trial 1 --dataset miniImage --cuda 0
python train.py  --model mobile_half_class_3 --arch_name Mobile_half_class_3_240_triple --trial 0 --dataset miniImage --cuda 1
python train.py  --model MobileNetV2 --arch_name MobileNetV2_mini_240_doconv --trial 24 --dataset miniImage --cuda 1
python train.py  --model mobile_half_class_3 --arch_name Mobile_half_class_3_240_doconv --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_norep --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_onlypw --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_onlydw --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_onlylinear --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_nolinear --trial 1 --dataset miniImage --cuda 1

明天
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_nodw --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_nopw --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+downsample --trial 1 --dataset miniImage --cuda 2
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+downsamplebn --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+changechannel --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+stem --trial 1 --dataset miniImage --cuda 2
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_doubledense_div --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_doubledense --trial 1 --dataset miniImage --cuda 1
python train.py  --model repvggnet --arch_name Repvggnet_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div2 --trial 1 --dataset miniImage --cuda 1
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div3 --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_dwnoact --trial 2 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_dwnoact_linearact --trial 2 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_new --trial 2 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_new_2 --trial 2 --dataset miniImage --cuda 1

McuNetv1
python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240 --trial 1 --dataset miniImage --cuda 1
python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_res160 --trial 1 --dataset miniImage --cuda 1

python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_25 --trial 2 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_both_doubledense_div --trial 2 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_both_doubledense_div_pwnoact --trial 2 --dataset miniImage --cuda 1

python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_08 --trial 1 --dataset miniImage --cuda 1

python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_identobn --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_dw --trial 1 --dataset miniImage --cuda 0
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_pw --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_linear --trial 1 --dataset miniImage --cuda 1
python train.py  --model RepTinynet25 --arch_name RepTinynet25_mini_240_+no_linear_stem_doubledense_div_06 --trial 1 --dataset miniImage --cuda 1

python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_inil --trial 1 --dataset miniImage --cuda 1
python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_inil_doubleclass --trial 1 --dataset miniImage --cuda 1
python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_inil_doubleclassbn --trial 1 --dataset miniImage --cuda 1


python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_origional_06 --trial 1 --dataset miniImage --cuda 1
python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_origional_08 --trial 1 --dataset miniImage --cuda 1
python train.py  --model mobile_half_class_3 --arch_name Mobile_half_class_3_240_doconvidv --trial 0 --dataset miniImage --cuda 0


python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_half_class --arch_name kd_mobile_half_class_init_mini --trial 1  -r 0.1 -a 0.9 -b 0  --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_half_class_3 --arch_name kd_mobile_half_class_3_init_mini --trial 1  -r 0.1 -a 0.9 -b 0  --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_half_class_3 --arch_name kd_mobile_half_class_3_init_mini_55 --trial 1  -r 0.5 -a 0.5 -b 0  --dataset miniImage
python train_student.py --path_t /root/distill/path/teacher_model/mobile_vit_small_init_miniImage_lr_0.01_decay_0.0005_trial_1/mobile_vit_small_init_best.pth --model_s mobile_half_class --arch_name kd_mobile_half_class_init_mini_55 --trial 1  -r 0.5 -a 0.5 -b 0  --dataset miniImage

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_weight111_div3 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_weight111_div2 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_onlydoubledense_div2 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_onlydoubledense_div3 --trial 0 --dataset miniImage --cuda 1
以上4个重做，因为23最佳的时候是only-double dense,没有idout的分支残差结构

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_stem1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw_onedense1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw1x1 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwlinear1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_class1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_classone1x1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_all1x1 --trial 0 --dataset miniImage --cuda 1


python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_onlydoubledense_test --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_weight11_div2 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_weight55 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_onlydoubledense_div2 --trial 0 --dataset miniImage --cuda 0


python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2 --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2_1x1 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2_double1x1 --trial 0 --dataset miniImage --cuda 0
79.8667


python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_stem_inver_nostride2 --trial 0 --dataset miniImage --cuda 1

python train.py  --model repvit_m0_6 --arch_name repvit_m0_6_init_mini_80 --trial 6 --dataset miniImage  --cuda 0 
python train.py  --model repvit_m0_6 --arch_name repvit_m0_6_init_mini_80_class --trial 6 --dataset miniImage  --cuda 0 


python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2_1x1_1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2_double1x1_1 --trial 0 --dataset miniImage --cuda 1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2_1x1_0 --trial 0 --dataset miniImage --cuda 0
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_doubledense_test_div2_double1x1_0 --trial 0 --dataset miniImage --cuda 0


python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_div2_inver_one1x1 --trial 0 --dataset miniImage --cuda 1
79.333
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_div2_inver_one1x1_0 --trial 0 --dataset miniImage --cuda 0
78.9083
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_nodiv2_inver_one1x1_0 --trial 0 --dataset miniImage --cuda 0
78.9833
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_nodiv2_inver_one1x1_0 --trial 0 --dataset miniImage --cuda 1
79.8167

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_div2_inver_both1x1 --trial 0 --dataset miniImage --cuda 1
79.4916
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_div2_inver_both1x1_0 --trial 0 --dataset miniImage --cuda 0
79.8499
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_nodiv2_inver_both1x1_0 --trial 0 --dataset miniImage --cuda 0
78.9833
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_nodiv2_inver_both1x1_1 --trial 0 --dataset miniImage --cuda 1
79.33

python train.py  --model mcunetlike --arch_name mcunetlike_mini_240 --trial 0 --dataset miniImage --cuda 1
77.20
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_0 --trial 0 --dataset miniImage --cuda 0

python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_simpleclass --trial 0 --dataset miniImage --cuda 1
79.06
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_simpleclass_0 --trial 0 --dataset miniImage --cuda 0


python train.py  --model mcunetlike --arch_name mcunetlike_mini_240 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_simpleclass --trial 0 --dataset miniImage --cuda 1
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_classdiv --trial 0 --dataset miniImage --cuda 0
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_classnodiv --trial 0 --dataset miniImage --cuda 0

python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_classdiv --trial 0 --dataset miniImage --cuda 0
78.5167
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_classdiv --trial 0 --dataset miniImage --cuda 1
78.6833
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_classdiv_bn --trial 0 --dataset miniImage --cuda 0
78.57499
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_classdiv_bn --trial 0 --dataset miniImage --cuda 1
78.10833
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dov_inver_nostride2_div2_inver_both1x1_expand4 --trial 0 --dataset miniImage --cuda 1
78.8417

python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_nodroup --trial 0 --dataset miniImage --cuda 0
78.1333
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_simple_another --trial 0 --dataset miniImage --cuda 0
77.9250
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_simple_first3 --trial 0 --dataset miniImage --cuda 1
79.2417
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_first3 --trial 0 --dataset miniImage --cuda 1
78.2333
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_simple_originalblocknum --trial 0 --dataset miniImage --cuda 1
78.5500
python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_originalblocknum --trial 0 --dataset miniImage --cuda 0
78.8833

python train.py  --model mcunetlike --arch_name mcunetlike_mini_240_class_simple_another_nobn --trial 0 --dataset miniImage --cuda 1

python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_impl_08_stem_dw_dilated --trial 1 --dataset miniImage --cuda 0
79.3500
python train.py  --model McuNetv1 --arch_name McuNetv1_mini_240_impl_08_stem_dw_dilated_div --trial 1 --dataset miniImage --cuda 1
78.9500

mobile_half
python train.py  --model MobileNetV2 --arch_name mobile_half_mini_240_hyper_original --trial 0 --dataset miniImage --cuda 1
77.133
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_410_pw --trial 0 --dataset miniImage --cuda 1
76.5999
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw --trial 0 --dataset miniImage --cuda 0
67.375
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_linear --trial 0 --dataset miniImage --cuda 0
74.9917
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_all --trial 0 --dataset miniImage --cuda 1
61.0167

---效果这么差，肯定是触及到梯度的更新了，加skip_connection呗
重跑之后：

python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_pw --trial 0 --dataset miniImage --cuda 1
77.8333--日志丢失了，重新跑下：77.2917
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw --trial 0 --dataset miniImage --cuda 0
78.1750
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_linear --trial 0 --dataset miniImage --cuda 0
76.7583
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_all --trial 0 --dataset miniImage --cuda 1
77.013
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw_pw --trial 0 --dataset miniImage --cuda 0
77.6333
--这里的groups好像写错了，重新做一下实验
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw——2 --trial 0 --dataset miniImage --cuda 1
78.0833---有效
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw_stem_3x3_only_dense --trial 0 --dataset miniImage --cuda 1
79.2000--这里没有后面的1x1不太行，但是不能rep注定原来的模型是不可行的，多并行lantency太高。
那做以下方案：
1.把1x1放到最后，只加一个小的ffc
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw_stem_1_3x3_only_dense --trial 0 --dataset miniImage --cuda 1
78.8667
2.为stem的分支池增加 Reptower
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw_stem_tower1_3x3_only_dense --trial 0 --dataset miniImage --cuda 0
78.8000
3. 在2的基础上为origin也增加1x1：
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw_stem_tower2_3x3_only_dense --trial 0 --dataset miniImage --cuda 0[]
79.6500
4. 在3的基础上为dw也加1x1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dw_stem_tower3_3x3_only_dense --trial 0 --dataset miniImage --cuda 1
79.3083



python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw——4 --trial 0 --dataset miniImage --cuda 0
linear不能加，那下面就是测试pw不加bn以及dw扩大的倍数。 77.4417
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_pw1x1_nobn --trial 0 --dataset miniImage --cuda 1
77.3083
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_pw1x1_only2nobn --trial 0 --dataset miniImage --cuda 1
77.0833
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_pwdw1x1_only2nobn --trial 0 --dataset miniImage --cuda 0
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw_3 --trial 0 --dataset miniImage --cuda 0
77.475


---效果不错，但是代码写乱了，用eval来测试是不是非修改之前的6-0.5--看模型框架是没问题的，该方法有效，切在dw的膨胀空间提升更高。
needle的rep验证代码在association.py
python train.py  --model MobileNetV2 --arch_name mobile_half_1_1_1_1_mini_240_dw_test --trial 0 --dataset miniImage --cuda 0

---5.09要跑的代码包括按照UniRepKLNet对dw进行修改，那其实就是加个SE模块了


pw dw都有, pw的expand上升到3
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower35_3x3_only_dense --trial 0 --dataset miniImage --cuda 0
78.6750
pw dw都有, pw的expand上升到3 且pw在stride=2时无纵向rep--比上面会好一些hh
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower30nostride2_3x3_only_dense --trial 0 --dataset miniImage --cuda 1
79.6833
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower30nostride2_3x3_only_dense --trial 0 --dataset miniImage --cuda 1

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_towernostride2__only_dense_dov --trial 0 --dataset miniImage --cuda 1
79.0333
pw在stride=2的时候dov和reptower都有
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov --trial 0 --dataset miniImage --cuda 0
78.1833

这里把上面的reptower去掉
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov——1 --trial 0 --dataset miniImage --cuda 0

pw也变成rep类型的
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov——2 --trial 0 --dataset miniImage --cuda 1

repvgg变传统的结构做对比
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov——3 --trial 0 --dataset miniImage --cuda 1
传统结构无1x1
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov——4 --trial 0 --dataset miniImage --cuda 0
传统结构无BN
python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov——5 --trial 0 --dataset miniImage --cuda 0

python train.py  --model RepTinynet23 --arch_name RepTinynet23_mini_240_dwpw_stem_tower_only_dense_dov——传统结构BN变identity --trial 0 --dataset miniImage --cuda 0

太乱了上面的。。。
重做了，因为原来的stem无法rep，所以重做有必要性，边做边写论文吧。
Constract1. stem 7的原始uni  接3x3卷积follow， pw原始， dw only——dense+div， linear原始， classifier ensemble
python train.py  --model ContrastNet1 --arch_name ContrastNet1 --trial 0 --dataset miniImage --cuda 1 --batch_size 56
79.3750
*80.0833
stem 7的原始un i  接1x1卷积follow， pw原始， dw only——dense+div， linear原始， classifier ensemble
python train.py  --model ContrastNet2 --arch_name ContrastNet2 --trial 0 --dataset miniImage --cuda 1 --batch_size 56
78.8833
*80.1000
80.433
stem 7的原始uni stem的uni增加tower，expand为4 接3x3卷积follow， pw原始， dw only——dense+div， linear原始， classifier ensemble
python train.py  --model ContrastNet3 --arch_name ContrastNet3 --trial 0 --dataset miniImage --cuda 0
78.5083
stem 7的原始uni stem的uni增加tower，expand为2 接3x3卷积follow， pw原始， dw only——dense+div， linear原始， classifier ensemble
python train.py  --model ContrastNet3 --arch_name ContrastNet3 --trial 0 --dataset miniImage --cuda 0
79.3250
*79.0417
根据这里的准确率和loss，决定stem到底采用1x1还是3X3。expand是采用2还是4
后面的实验包括tower在inverted模块中的使用，pw--rep的增强，dov的使用以及针对下采样层时的应用。
首先我们尝试全部增加，先全部增加dov，再全部增加tower，再dov和tower全部增加，此时不区分是否为stride=2

stem在expand=2的情况下增加dov
python train.py  --model ContrastNet4 --arch_name ContrastNet4 --trial 0 --dataset miniImage --cuda 1
79.70
*79.6833
3的底子，expand=2全部增加tower 这里inverted的linear降维层没加---load2
python train.py  --model ContrastNet5 --arch_name ContrastNet5 --trial 0 --dataset miniImage --cuda 0
*79.3750
3的底子，expand=2全部增加dov---这里只是为inverted全加dov，且stride=2时，pw层不加dov  -load在跑
python train.py  --model ContrastNet6 --arch_name ContrastNet6 --trial 0 --dataset miniImage --cuda 0
76.1667
*78.9750
3的底子，expand=2全部增加dov+tower  -load在跑
python train.py  --model ContrastNet7 --arch_name ContrastNet7 --trial 0 --dataset miniImage --cuda 0
78.2667
*78.3583
7的底子stride=2时，不加tower--在跑
python train.py  --model ContrastNet8 --arch_name ContrastNet8 --trial 0 --dataset miniImage --cuda 0
79.2250
*78.341667175
8的底子,对pw也是用并行rep  ---load1跑
python train.py  --model ContrastNet9 --arch_name ContrastNet9 --trial 0 --dataset miniImage --cuda 0
77.6999
*79.9083
9的底子,改为原始并行rep--但加div--去load1跑
python train.py  --model ContrastNet10 --arch_name ContrastNet10 --trial 0 --dataset miniImage --cuda 0
77.0916
*76.8667
**77.5500
10的底子，但rep加tower和dov--load1跑
python train.py  --model ContrastNet11 --arch_name ContrastNet11 --trial 0 --dataset miniImage --cuda 0
74.9500
batch增大后
76.6333
修改后
77.0167
*76.7583
单卡跑77.0250--size 104
11的底子，并行BN换identity--load2
python train.py  --model ContrastNet12 --arch_name ContrastNet12 --trial 0 --dataset miniImage --cuda 1
本机跑的 batch_size=104  78.1917
*77.9333


python train.py  --model mobile_half_class --arch_name mobile_class_both --trial 0 --dataset miniImage --cuda 1 --batch_size 56 
python train.py  --model mobile_half_class --arch_name mobile_class_nobn --trial 0 --dataset miniImage --cuda 1 --batch_size 56 
python train.py  --model mobile_half_class --arch_name mobile_class_nobndrop --trial 0 --dataset miniImage --cuda 1 --batch_size 56 

python train.py  --model mobile_half_class --arch_name mobile_class_both_128 --trial 0 --dataset miniImage --cuda 1 
python train.py  --model mobile_half_class --arch_name mobile_class_nobn_128 --trial 0 --dataset miniImage --cuda 1 
python train.py  --model mobile_half_class --arch_name mobile_class_nobndrop_128 --trial 0 --dataset miniImage --cuda 1


听从了徐老师的建议，那就全使用128的batch_size。 在本服务器上的 ContrastNet2_nobndrop 得到了80.6833的准确率，loss为0.71729
整体重新做一次实验，将loss和weight保存下来。
1. 首先构建baseline信息，在load服务器上训练3个同样的baseline模型，取中间准确率的为基准模型。
python train.py  --model MobileNetV2 --arch_name Baseline1 --trial 0 --dataset miniImage --cuda 1
77.983329
python train.py  --model MobileNetV2 --arch_name Baseline2 --trial 1 --dataset miniImage --cuda 2
77.400
python train.py  --model MobileNetV2 --arch_name Baseline3 --trial 2 --dataset miniImage --cuda 3
78.0999
2. 根据4个点，分别验证他们的有效性，包括stem大卷积， 主干repdiv， ensemble分类器和 vertical rep
 -1， 首先验证ensemble分类器，在load服务器上跑class和class2, load_1 服务器上跑class3。为了验证class处div的重要性，分别对class2和class3进行nodiv处理。

self.add_module('drop', torch.nn.Dropout(0.2))
self.add_module('l', torch.nn.Linear(a, b, bias=bias))
python train.py  --model mobile_half_class --arch_name Class1_1 --trial 0 --dataset miniImage --cuda 4
78.8917
python train.py  --model mobile_half_class --arch_name Class1_2 --trial 1 --dataset miniImage --cuda 5
79.1000
python train.py  --model mobile_half_class --arch_name Class1_3 --trial 2 --dataset miniImage --cuda 6
79.6167

无drop时：load_
python train.py  --model mobile_half_class --arch_name Class1_1 --trial 3 --dataset miniImage --cuda 1
79.0667
python train.py  --model mobile_half_class --arch_name Class1_2 --trial 4 --dataset miniImage --cuda 2
78.5833
python train.py  --model mobile_half_class --arch_name Class1_3 --trial 5 --dataset miniImage --cuda 3
79.0917

python train.py  --model mobile_half_class_2 --arch_name Class2_1 --trial 0 --dataset miniImage --cuda 7
78.9333
python train.py  --model mobile_half_class_2 --arch_name Class2_2 --trial 1 --dataset miniImage --cuda 8
78.77500
python train.py  --model mobile_half_class_2 --arch_name Class2_3 --trial 2 --dataset miniImage --cuda 9
78.2166

python train.py  --model mobile_half_class_3 --arch_name Class3_1 --trial 0 --dataset miniImage --cuda 1
79.8917
python train.py  --model mobile_half_class_3 --arch_name Class3_2 --trial 1 --dataset miniImage --cuda 8
79.8667
python train.py  --model mobile_half_class_3 --arch_name Class3_3 --trial 2 --dataset miniImage --cuda 4
79.2667
从以上对比实验可以看出class的有效性，同时drop不可丢弃。
    
python train.py  --model mobile_half_class_2 --arch_name Class2_1_nodiv --trial 3 --dataset miniImage --cuda 0
78.5500
python train.py  --model mobile_half_class_2 --arch_name Class2_2_nodiv --trial 4 --dataset miniImage --cuda 0
78.1500
python train.py  --model mobile_half_class_2 --arch_name Class2_3_nodiv --trial 5 --dataset miniImage --cuda 0

python train.py  --model mobile_half_class_3 --arch_name Class3_1_nodiv --trial 3 --dataset miniImage --cuda 1
78.5917
python train.py  --model mobile_half_class_3 --arch_name Class3_2_nodiv --trial 4 --dataset miniImage --cuda 1
78.6833
python train.py  --model mobile_half_class_3 --arch_name Class3_3_nodiv --trial 5 --dataset miniImage --cuda 1
从以上实验可以看出div对最终结果的影响显著。

对class进行迁移，shufflenet load_， resnet load_1, mobilenetv1 load_1
ShuffleNetv1:
python train.py  --model ShuffleV1 --arch_name shuffleV1 --trial 0 --dataset miniImage --cuda 4
75.1167
python train.py  --model ShuffleV1 --arch_name shuffleV1_1 --trial 0 --dataset miniImage --cuda 5
74.6333
下两个实际为3，有dropout
python train.py  --model ShuffleV1 --arch_name shuffleV1_class --trial 0 --dataset miniImage --cuda 6
77.1583
python train.py  --model ShuffleV1 --arch_name shuffleV1_1_class --trial 0 --dataset miniImage --cuda 7
75.824996
python train.py  --model ShuffleV1 --arch_name shuffleV1_class2 --trial 0 --dataset miniImage --cuda 8
76.5583
python train.py  --model ShuffleV1 --arch_name shuffleV1_1_class2 --trial 0 --dataset miniImage --cuda 9
76.7167

ResNet:
python train.py  --model ResNet18 --arch_name ResNet18 --trial 0 --dataset miniImage --cuda 1
81.8750
python train.py  --model ResNet18 --arch_name ResNet18_class2 --trial 1 --dataset miniImage --cuda 2
81.7250
python train.py  --model ResNet18 --arch_name ResNet18_class3 --trial 2 --dataset miniImage --cuda 3
82.1250


python train.py  --model ResNet50 --arch_name ResNet50 --trial 0 --dataset miniImage --cuda 4
81.5500
python train.py  --model ResNet50 --arch_name ResNet50_class2 --trial 1 --dataset miniImage --cuda 5
84.1417
python train.py  --model ResNet50 --arch_name ResNet50_class3 --trial 2 --dataset miniImage --cuda 6
84.5250

其次是分析stem的大卷积，首先基于基础模型进行分析，原本是3x3的卷积，可在其前面增加一个3-3通道不变的大卷积用于特征提取和融合-后使用1x1卷积来进行升维。或采用类似metaformer的思想。大卷积变成rep类型的。亦或就简单地在其前面加一个rep或非rep的大卷积提取特征。3x3不变。
在load2进行：最后结果是先2后1接3x3 act非group或group随意stride，但是是both
原版uni后面跟bn，se，ffc，act。其中se和ffc不考虑。那就是每组做2个后面接bn或act或bn act或都不接。

1.3x3不变，前面加group的rep3-3通道映射的uni_ load_1  这里暂时推荐先2后一 both 78.2583
 本服务器和load_1都跑一次---本机跑概率高是因为调的是缩小8倍的--cifar的那组参数
 1.单bn
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_bn1 --trial 0 --dataset miniImage --cuda 8   实验室服务器
 77.1917
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_bn2 --trial 1 --dataset miniImage --cuda 7  实验室服务器
76.8667
    2.单act
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_act1 --trial 2 --dataset miniImage --cuda 6  实验室服务器
77.8833
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_act2 --trial 3 --dataset miniImage --cuda 5 实验室服务器
77.1667
    3.全有
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_both1 --trial 4 --dataset miniImage --cuda 4 0实验室服务器 
    77.5250
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_both2 --trial 5 --dataset miniImage --cuda 3 0实验室服务器
    78.2583

    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_both2 --trial 5 --dataset miniImage --cuda 1
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_both2 --trial 5 --dataset miniImage --cuda 2
    4.都无
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_no1 --trial 6 --dataset miniImage --cuda 1 
    76.7417
    python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_no2 --trial 7 --dataset miniImage --cuda 2
    76.8667
python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_no_changestride --trial 1 --dataset miniImage --cuda 1
77.8083
python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_act_changestride --trial 2 --dataset miniImage --cuda 2
77.6717
python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_norm_changestride --trial 3 --dataset miniImage --cuda 3
77.3167
python train.py  --model mobile_half_1_1_1_2 --arch_name mobile_half_1_1_1_2_both_changestride --trial 4 --dataset miniImage --cuda 4
77.5250

2.3x3不变，前面加非group3-3通道映射的uni load_2 --目测是act或都加效果最好
mobile_half_1_1_1_1  78.2250 act  这里推荐先2后一act
    1.单bn
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_bn1 --trial 0 --dataset miniImage --cuda 1
    77.2833  
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_bn2 --trial 1 --dataset miniImage --cuda 2 
    77.1500
    2.单act
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_act1 --trial 2 --dataset miniImage --cuda 3  
    77.8750
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_act2 --trial 3 --dataset miniImage --cuda 4
    78.2250
    3.全有
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_both1 --trial 4 --dataset miniImage --cuda 5
    78.1083  
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_both2 --trial 5 --dataset miniImage --cuda 6
    78.0250
    4.都无
    python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_no1 --trial 6 --dataset miniImage --cuda 0
    77.7417 
    # python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_no2 --trial 7 --dataset miniImage --cuda 1 本服务器--比例错了
stride 2-1置换之后  77.6917 both

python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_no_changestride --trial 7 --dataset miniImage --cuda 1
76.7667
python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_act_changestride --trial 8 --dataset miniImage --cuda 2
77.2083
python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_norm_changestride --trial 9 --dataset miniImage --cuda 3
77.6250
python train.py  --model mobile_half_1_1_1_1 --arch_name mobile_half_1_1_1_1_both_changestride --trial 10 --dataset miniImage --cuda 4
77.6917

3.3x3变为1x1，前面加非group的rep的3-3通道映射的uni  load_1  这里推荐先2后一，both--78.200全有hh
      1.单bn
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_bn1 --trial 0 --dataset miniImage --cuda 3
  77.600
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_bn2 --trial 1 --dataset miniImage --cuda 4
  77.500
    2.单act
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_act1 --trial 2 --dataset miniImage --cuda 5
  77.5750
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_act2 --trial 3 --dataset miniImage --cuda 6
  77.875
    3.全有
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both1 --trial 4 --dataset miniImage --cuda 7 
    78.200
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both2 --trial 5 --dataset miniImage --cuda 8
    77.3250
    4.都无
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_no1 --trial 6 --dataset miniImage --cuda 9 
    77.9583
    python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_no2 --trial 7 --dataset miniImage --cuda 0
    77.7167
  stride 对换  --load——2
  
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_no_changestride --trial 1 --dataset miniImage --cuda 1
77.8083
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_act_changestride --trial 2 --dataset miniImage --cuda 2
77.6717
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_norm_changestride --trial 3 --dataset miniImage --cuda 3
77.3167
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both_changestride --trial 4 --dataset miniImage --cuda 4
77.5250

--换大核
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both_kernel9_1 --trial 11 --dataset miniImage --cuda 1
76.9750
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both_kernel9_2 --trial 12 --dataset miniImage --cuda 2
77.5583

python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both_kernel11_1 --trial 13 --dataset miniImage --cuda 3
77.3917
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both_kernel11_2 --trial 14 --dataset miniImage --cuda 4
77.5417
python train.py  --model mobile_half_1_1_1 --arch_name mobile_half_1_1_1_both_kernel7_3 --trial 20 --dataset miniImage --cuda 5 load2
77.6667

4.3x3变为1x1，前面加group的3-3通道映射的uni load_  --78.025 BN很重要  这里推荐先1后2，both
 1.单bn
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_bn1 --trial 0 --dataset miniImage --cuda 3
 77.7000
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_bn2 --trial 1 --dataset miniImage --cuda 4
77.7167
    2.单act
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_act1 --trial 2 --dataset miniImage --cuda 5
  77.1583
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_act2 --trial 3 --dataset miniImage --cuda 6
 77.2667
    3.全有
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_both1 --trial 4 --dataset miniImage --cuda 7 
77.625
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_both2 --trial 5 --dataset miniImage --cuda 8
77.6250
    4.都无
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_no1 --trial 6 --dataset miniImage --cuda 9 
    77.1750
    python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_no2 --trial 7 --dataset miniImage --cuda 0
    76.5750
stride - 12对换  78.0250
python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_no_changestride --trial 11 --dataset miniImage --cuda 1
76.7667
python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_act_changestride --trial 8 --dataset miniImage --cuda 2
76.9667
python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_norm_changestride --trial 9 --dataset miniImage --cuda 3
77.0333
python train.py  --model mobile_half_1_1_2 --arch_name mobile_half_1_1_2_both_changestride --trial 10 --dataset miniImage --cuda 4
78.0250

这里尝试把uni的kernel扩大。如9 11 13




vertical rep
最初的stem，无效
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_stem_1 --trial 0 --dataset miniImage --cuda 1
77.3667
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_stem_2 --trial 1 --dataset miniImage --cuda 2
77.4333

这里的pw后面的tower没有用group，group的结果看load_1
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_dw_3 --trial 2 --dataset miniImage --cuda 3

python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_dw_4 --trial 3 --dataset miniImage --cuda 4



python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwboth_5 --trial 4 --dataset miniImage --cuda 5
77.2500
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwboth_6 --trial 5 --dataset miniImage --cuda 6
77.0583

python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_linear_dw_1 --trial 12 --dataset miniImage --cuda 1
77.1083
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_linear_dw_2 --trial 13 --dataset miniImage --cuda 2
77.0333
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwstride2 --trial 14 --dataset miniImage --cuda 3
77.9833
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwstride2_1 --trial 15 --dataset miniImage --cuda 4
77.6833
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwnostride2 --trial 16 --dataset miniImage --cuda 5
77.2750
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwnostride2_1 --trial 17 --dataset miniImage --cuda 6
77.7750


python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwnostride2_dw --trial 18 --dataset miniImage --cuda 1
77.7083
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwnostride2_dw_1 --trial 19 --dataset miniImage --cuda 2
77.8667
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwnostride2_dw_stride2_dw --trial 20 --dataset miniImage --cuda 3
77.3250
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwnostride2_dw_stride2_dw_1 --trial 21 --dataset miniImage --cuda 4
77.4583
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwstride2_dw --trial 22 --dataset miniImage --cuda 5
**78.4083
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwstride2_dw_1 --trial 23 --dataset miniImage --cuda 6
76.7750
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwstride2_dw_nostride2_dw --trial 24 --dataset miniImage --cuda 7
77.5833
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwstride2_dw_nostride2_dw_1 --trial 25 --dataset miniImage --cuda 8
77.2583


python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_linear_1 --trial 6 --dataset miniImage --cuda 3
77.3083
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_linear_2 --trial 7 --dataset miniImage --cuda 2
77.2000

这里group设置错误了，重做
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwboth_5 --trial 4 --dataset miniImage --cuda 6
77.8083
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pwboth_6 --trial 5 --dataset miniImage --cuda 5
77.3750

python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_linear_dw_1 --trial 8 --dataset miniImage --cuda 7
77.1667
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_linear_dw_2 --trial 9 --dataset miniImage --cuda 8
77.2750
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pw_dw_1 --trial 10 --dataset miniImage --cuda 9
77.4833
python train.py  --model mobile_half_1_1_3 --arch_name mobile_half_1_1_3_pw_dw_2 --trial 11 --dataset miniImage --cuda 0
77.4250


finalNet
python train.py  --model finalNet --arch_name finalNet_1 --trial 0 --dataset miniImage --cuda 1  --batch_size 56
79.7417
python train.py  --model finalNet --arch_name finalNet_2 --trial 1 --dataset miniImage --cuda 0
79.5833
python train.py  --model finalNet --arch_name finalNet_1_56 --trial 3 --dataset miniImage --cuda 1  --batch_size 56


python train.py  --model ContrastNet2 --arch_name ContrastNet2_128 --trial 0 --dataset miniImage --cuda 1
79.6667
python train.py  --model ContrastNet2 --arch_name ContrastNet2_128_normstem --trial 0 --dataset miniImage --cuda 0
79.2917
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_normstem --trial 0 --dataset miniImage --cuda 0  --batch_size 56
80.4250
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_normstem_clasnodrop --trial 2 --dataset miniImage --cuda 0  --batch_size 56
80.4250

python train.py  --model ContrastNet2 --arch_name ContrastNet2_56 --trial 1 --dataset miniImage --cuda 1  --batch_size 56
80.9167
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_classnodrop --trial 3 --dataset miniImage --cuda 1  --batch_size 56
80.7917

基于这个80.9167来进行后续的实验。
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_baseline --trial 4 --dataset miniImage --cuda 1  --batch_size 56
79.7999954
为主干dw和stride不等于2时的pw加tower
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_tower --trial 5 --dataset miniImage --cuda 1  --batch_size 56
80.8167
为主干dw和stride不等于2时的pw加tower，为stem加tower
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_tower_plus --trial 6 --dataset miniImage --cuda 0 --batch_size 56
80.1250
为主干dw和stride不等于2时的pw加上下tower
ython train.py  --model ContrastNet2 --arch_name ContrastNet2_56_updowntower --trial 7 --dataset miniImage --cuda 0  --batch_size 56
80.6917
constrastNet2+stem的tower
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_stem_tower --trial 8 --dataset miniImage --cuda 1  --batch_size 56
80.6833

tower单单升降,二层结构
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_stem_tower_nohigh --trial 9 --dataset miniImage --cuda 1  --batch_size 56
80.9833
还有就是pw用并行的rep
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_tower_nohigh --trial 5 --dataset miniImage --cuda 1  --batch_size 56
80.3250
单层结构tower_dw和nostride的pw
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_single_tower --trial 9 --dataset miniImage --cuda 1  --batch_size 56
80.2417
tower 二层结构 rate变3
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_stem_tower_nohigh——3 --trial 10 --dataset miniImage --cuda 1  --batch_size 56
80.9417
+哥pwnostride2
python train.py  --model ContrastNet2 --arch_name ContrastNet2_56_stem_tower_nohigh——4 --trial 11 --dataset miniImage --cuda 0  --batch_size 56
80.2833
div直接套用stem和repvgg的实验以及以下dw和原始rep模块的对比即可


python train.py  --model ContrastNet2 --arch_name ContrastNet2_11 --trial 99 --dataset miniImage --cuda 1
77.8833  
python train.py  --model ContrastNet2 --arch_name ContrastNet2_13 --trial 98 --dataset miniImage --cuda 1  
78.1500
python train.py  --model ContrastNet2 --arch_name ContrastNet2_9 --trial 97 --dataset miniImage --cuda 0
77.3583
python train.py  --model ContrastNet2 --arch_name ContrastNet2_7 --trial 96 --dataset miniImage --cuda 0  
78.0083
python train.py  --model ContrastNet2 --arch_name ContrastNet2_5 --trial 95 --dataset miniImage --cuda 1  
77.9667
python train.py  --model ContrastNet2 --arch_name ContrastNet2_3 --trial 94 --dataset miniImage --cuda 1  
77.775
python train.py  --model ContrastNet2 --arch_name ContrastNet2_no --trial 93 --dataset miniImage --cuda 0  
77.0750

python train.py  --model ContrastNet2 --arch_name ContrastNet2_11_3 --trial 99 --dataset miniImage --cuda 1
 
python train.py  --model ContrastNet2 --arch_name ContrastNet2_13_3 --trial 98 --dataset miniImage --cuda 1  

python train.py  --model ContrastNet2 --arch_name ContrastNet2_9_3 --trial 97 --dataset miniImage --cuda 0

python train.py  --model ContrastNet2 --arch_name ContrastNet2_7_3 --trial 96 --dataset miniImage --cuda 0  

python train.py  --model ContrastNet2 --arch_name ContrastNet2_5_3 --trial 95 --dataset miniImage --cuda 1  

python train.py  --model ContrastNet2 --arch_name ContrastNet2_3_3 --trial 94 --dataset miniImage --cuda 1  


python train.py  --model ContrastNet2 --arch_name ContrastNet2_11_cifar_1 --trial 92 --dataset cifar100 --cuda 1
66.0300
python train.py  --model ContrastNet2 --arch_name ContrastNet2_13_cifar_1 --trial 91 --dataset cifar100 --cuda 1  
65.7700
python train.py  --model ContrastNet2 --arch_name ContrastNet2_9_cifar_1 --trial 90 --dataset cifar100 --cuda 0
65.9700
python train.py  --model ContrastNet2 --arch_name ContrastNet2_7_cifar_1 --trial 89 --dataset cifar100 --cuda 0  
66
python train.py  --model ContrastNet2 --arch_name ContrastNet2_5_cifar_1 --trial 88 --dataset cifar100 --cuda 1  
66
python train.py  --model ContrastNet2 --arch_name ContrastNet2_3_cifar_1 --trial 87 --dataset cifar100 --cuda 1  
65.75
python train.py  --model ContrastNet2 --arch_name ContrastNet2_no_cifar --trial 86 --dataset cifar100 --cuda 0 
67.77


python train.py  --model ContrastNet2 --arch_name ContrastNet2_11_cifar_3 --trial 85 --dataset cifar100 --cuda 1
66.12
python train.py  --model ContrastNet2 --arch_name ContrastNet2_13_cifar_3 --trial 84 --dataset cifar100 --cuda 1  
66.20
python train.py  --model ContrastNet2 --arch_name ContrastNet2_9_cifar_3 --trial 83 --dataset cifar100 --cuda 0
65.69
python train.py  --model ContrastNet2 --arch_name ContrastNet2_7_cifar_3 --trial 82 --dataset cifar100 --cuda 0  
66.70
python train.py  --model ContrastNet2 --arch_name ContrastNet2_5_cifar_3 --trial 81 --dataset cifar100 --cuda 1  
65.900
python train.py  --model ContrastNet2 --arch_name ContrastNet2_3_cifar_3 --trial 80 --dataset cifar100 --cuda 1
66.0900

没用不知道为啥，明天再看。

完全再load_跑的测试tower对降低expand——rate的帮助--降低到2，将tower放到stem， dw，pw不同的组合。
python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_ture --trial 10 --dataset miniImage --cuda 4 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_dw_ture --trial 15 --dataset miniImage --cuda 5 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_dw_pw_ture --trial 16 --dataset miniImage --cuda 6 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_dw_pwnostride2_ture --trial 17 --dataset miniImage --cuda 7 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_pwnostride2_ture --trial 18 --dataset miniImage --cuda 8 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_pw_ture --trial 19 --dataset miniImage --cuda 9 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_stem_ture --trial 11 --dataset miniImage --cuda 3 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_stem_dw_ture --trial 14 --dataset miniImage --cuda 4 --batch_size 56


python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_stem_dw_pw_ture --trial 12 --dataset miniImage --cuda 2 --batch_size 56

python train.py  --model ContrastNet2 --arch_name ContrastNet2——inil_expand_rate_2_stem_dw_pwno2_ture --trial 13 --dataset miniImage --cuda 1 --batch_size 56


python train.py  --model repvggnet --arch_name Repvggnet_mini_240 --trial 1 --dataset miniImage --cuda 1
79.324
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div2 --trial 1 --dataset miniImage --cuda 1
79.1166 这里loss是比上面小的，给机会再重新做一次。
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div2——2 --trial 1 --dataset miniImage --cuda 1
79.6416
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div3 --trial 1 --dataset miniImage --cuda 1
80.099

python train.py  --model repvggnet --arch_name Repvggnet_mini_240_endbn --trial 2 --dataset miniImage --cuda 0
return self.bnend(self.nonlinearity(self.se((self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))))
79.8167
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_noparallelbn --trial 3 --dataset miniImage --cuda 1
78.766


python train.py  --model repvggnet --arch_name Repvggnet_mini_240cifar --trial 4 --dataset cifar100 --cuda 1
69.7500
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div2cifar --trial 5 --dataset cifar100 --cuda 1
69.8400


python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div3cifar --trial 6 --dataset cifar100 --cuda 1
70.3600

python train.py  --model repvggnet --arch_name Repvggnet_mini_240_endbncifar --trial 7 --dataset cifar100 --cuda 0
return self.bnend(self.nonlinearity(self.se((self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))))
70.2600
python train.py  --model repvggnet --arch_name Repvggnet_mini_240_noparallelbncifar --trial 8 --dataset cifar100 --cuda 0
69.3500


这里是stem 的stride 1 2 ，不加act和norm，追cifar的指导来迁移imagenet。

python train.py  --model finalNet --arch_name finalNet_1 --trial 0 --dataset miniImage --cuda 1  --batch_size 56
77.4833
python train.py  --model finalNet --arch_name finalNet_1_two_stage_tower --trial 2 --dataset miniImage --cuda 1  --batch_size 56
78.0417
---act 还是很重要的
python train.py  --model finalNet --arch_name finalNet_1_two_stage_tower_128 --trial 1 --dataset miniImage --cuda 0  

python train.py  --model finalNet --arch_name finalNet_1_128_1 --trial 5 --dataset miniImage --cuda 0
79.9083

换tanh没用
python train.py  --model finalNet --arch_name finalNet_1_128_1 --trial 5 --dataset miniImage --cuda 0
78.8083
python train.py  --model finalNet --arch_name finalNet_1_128_1 --trial 4 --dataset miniImage --cuda 1 --batch_size 56  
79.56

相较于上者没有用tanh，rep没有identity,doconv也没有使用.
python train.py  --model finalNet --arch_name finalNet_2_128_1 --trial 5 --dataset miniImage --cuda 0
79.77
python train.py  --model finalNet --arch_name finalNet_2_128_2 --trial 7 --dataset miniImage --cuda 0
79.5
python train.py  --model finalNet --arch_name finalNet_2_56 --trial 4 --dataset miniImage --cuda 1 --batch_size 56  
79.73
rep里换leakrelu
python train.py  --model finalNet --arch_name finalNet_2_56——1 --trial 6 --dataset miniImage --cuda 1 --batch_size 56  
79.358

python train.py  --model mobile_half_class_3 --arch_name mobile_half_class_3_test --trial 99 --dataset miniImage --cuda 1 --epochs 1

cifar
python train.py  --model mobile_half_class_3 --arch_name mobile_half_class_3_test --trial 98 --dataset cifar100 --cuda 1 --epochs 1

python eval.py --model mobile_half_class_3 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/mobile_half_class_3_miniImage_lr_0.01_decay_0.0005_trial_99/mobile_half_class_3_best.pth

python eval.py --model mobile_half_class_3 --arch_name exp1_1 --dataset cifar100 --weights /root/distill/path/teacher_model/mobile_half_class_3_miniImage_lr_0.01_decay_0.0005_trial_98/mobile_half_class_3_best.pth

python train.py  --model MobileNetV2 --arch_name MobileNetV2_test --trial 101 --dataset cifar100 --cuda 1 --epochs 1
python eval.py --model MobileNetV2 --arch_name MobileNetV2_exp1 --dataset cifar100 --weights /root/distill/path/teacher_model/MobileNetV2_cifar100_lr_0.01_decay_0.0005_trial_101/MobileNetV2_best.pth
7944.133718452578
python train.py  --model MobileNetV2 --arch_name MobileNetV2_test --trial 101 --dataset miniImage --cuda 1 --epochs 1
python eval.py --model MobileNetV2 --arch_name MobileNetV2_exp2 --dataset miniImage --weights /root/distill/path/teacher_model/MobileNetV2_miniImage_lr_0.01_decay_0.0005_trial_101/MobileNetV2_best.pth


这里是expand_rate=2 stem 12 接tower， 接3x3， dw接tower ， pw stride=2不加tower tower+只进入bn tower 3层 pw 4 普通2: tower空的，就是不用tower
python train.py  --model finalNet --arch_name final——20 --trial 40 --dataset miniImage --cuda 0  --batch_size 56
79.0167
python train.py  --model finalNet --arch_name final——20—1 --trial 39 --dataset miniImage --cuda 1 --batch_size 56
79.2333

python train.py  --model finalNet --arch_name final——20-128 --trial 41 --dataset miniImage --cuda 0  
78.2417
python train.py  --model finalNet --arch_name final——20—1-128 --trial 42 --dataset miniImage --cuda 1 
78.8833


python train.py  --model finalNet --arch_name final——21 --trial 43 --dataset miniImage --cuda 0  --batch_size 56
80.5833
python train.py  --model finalNet --arch_name final——21—1 --trial 44 --dataset miniImage --cuda 1 --batch_size 56
81.0667

上面的基础上，扩大tower的rate为4
python train.py  --model finalNet --arch_name final——22 --trial 45 --dataset miniImage --cuda 0  --batch_size 56
80.5833
python train.py  --model finalNet --arch_name final——22—1 --trial 46 --dataset miniImage --cuda 1 --batch_size 56
80.3833

dw pw都是ddb，就全是ddb，但是分支都加了tower
python train.py  --model finalNet --arch_name final——23 --trial 47 --dataset miniImage --cuda 1 --batch_size 56
80.5
python eval.py --model finalNet --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_47/finalNet_best.pth



pw换成了单一的加tower
python train.py  --model finalNet --arch_name final——24 --trial 48 --dataset miniImage --cuda 0 --batch_size 56
79.9667

后面还能试一下我们的duplicate的

可以测试下tower能不能加模型的多样性嘛，做个对比实验，一个是4通道duplicate，一个是original+和后接1 2 3层的tower
这里先测试dw的变化，pw不变，stem不变
python train.py  --model finalNet --arch_name final——25 --trial 49 --dataset miniImage --cuda 1 --batch_size 56
80.4583
baseline就是后面不加tower
python train.py  --model finalNet --arch_name final——25——1 --trial 50 --dataset miniImage --cuda 0 --batch_size 56
80.633

后面的对比实验就是pw也加进来。--这里是加bn的
python train.py  --model finalNet --arch_name final——25——2 --trial 51 --dataset miniImage --cuda 1 --batch_size 56
80.9750
dw，pw 3duplicate, 不加tower
python train.py  --model finalNet --arch_name final——25——3 --trial 52 --dataset miniImage --cuda 0 --batch_size 56
81.4083

python eval.py --model finalNet --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_52/finalNet_best.pth
在81.4083的基础上于rep后添加tower
python train.py  --model finalNet --arch_name final——27 --trial 54 --dataset miniImage --cuda 1 --batch_size 56
81.2250
猜测会不会是深度可分离卷积并不适配这种映射，所以只在pw后面接试一试。
python train.py  --model finalNet --arch_name final——29 --trial 55 --dataset miniImage --cuda 0 --batch_size 56
81.0083
rep 4duplicate， pw dw都 rep. tower接 0123, div是sqrt分支数，没加大卷积stem
python train.py  --model finalNet --arch_name final——31 --trial 57 --dataset miniImage --cuda 0 --batch_size 56
81.4583
加上大卷积
python train.py  --model finalNet --arch_name final——33 --trial 59 --dataset miniImage --cuda 0 --batch_size 56
81.1333

换一个角度，如果是我用tower对他先升维呢
python train.py  --model finalNet --arch_name final——34 --trial 60 --dataset miniImage --cuda 1 --batch_size 56
81.1833

如果是考虑到bn不妨到降维层呢？这里tower的skipout也除以了2
python train.py  --model finalNet --arch_name final——32 --trial 58 --dataset miniImage --cuda 1 --batch_size 56
81.3583

后者尝试dw不加，pw加tower的123
python train.py  --model finalNet --arch_name final——30 --trial 56 --dataset miniImage --cuda 1 --batch_size 56
81.2417

在best的基础上，考虑dbb的有效性，可考虑残差结构存在的必要性，duplicate个数的对等性，pw升维duplicate的有效性
先将其提升到4duplicate，但dw没有用rep，只是接了tower
python train.py  --model finalNet --arch_name final——26 --trial 53 --dataset miniImage --cuda 0 --batch_size 56
80.9917

好奇一个问题，reptower，对残差结构使用会怎样？--我这里可以不用bn，这里如果work的话，那是一个巨大的突破。--这个实验没有做，训练后无法合并
python train.py  --model finalNet --arch_name final——28 --trial 54 --dataset miniImage --cuda 1 --batch_size 56

还有就是div的那个除数到底怎么取。

传统的skip_out用div有用吗？

做：基础的，没分支，没div，有div，查看他们 bnweight最后的值
从图来看，取倒数第三层block的dw最终的bn输出查看分布可知，多分枝结构和原模型具有相似的数据分布。当然，现在的结论是从rep2得到的，我们下面的示例是rep4.

python train.py  --model finalNet --arch_name final——35 --trial 61 --dataset miniImage --cuda 0 --batch_size 56
79.6250
python train.py  --model finalNet --arch_name final——36 --trial 62 --dataset miniImage --cuda 0 --batch_size 56
80.3167
python train.py  --model finalNet --arch_name final——37 --trial 63 --dataset miniImage --cuda 0 --batch_size 56
81.2750

5duplicate 00123  pw duplicate  dw ddb   /5
python train.py  --model finalNet --arch_name final——38 --trial 64 --dataset miniImage --cuda 1 --batch_size 56 
81.5250
5duplicate 00123  pw duplicate  dw ddb   /5 无tower
python train.py  --model finalNet --arch_name final——37 --trial 63 --dataset miniImage --cuda 0 --batch_size 56 --restart True --start_epochs 140
81.300

5duplicate 00123  pw duplicate  dw ddb   /5 stem /sqrt(5)
python train.py  --model finalNet --arch_name final——39 --trial 65 --dataset miniImage --cuda 1 --batch_size 56 
81.5500

5duplicate 00123  pw duplicate  dw ddb   /5 stem /sqrt(5)  tower bn放中间--这里stem的tower前置了
python train.py  --model finalNet --arch_name final——40 --trial 66 --dataset miniImage --cuda 0 --batch_size 56 
81.7667

python train.py  --model finalNet --arch_name final——41 --trial 67 --dataset miniImage --cuda 0 --batch_size 56
81.3833
dw 和pw都 duplicate， 参考load——的实验结果
python train.py  --model finalNet --arch_name final——42 --trial 68 --dataset miniImage --cuda 1 --batch_size 56
80.7833

测试mobileVit性能
mobile_vit_xx_small_in7
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7 --trial 0 --dataset miniImage --cuda 1
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7 --trial 0 --dataset miniImage --cuda 1
76.74
+stem tower 前置  重跑
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7_1 --trial 1 --dataset miniImage --cuda 0
76.88
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7_5 --trial 1 --dataset miniImage --cuda 0
77.04
+stem tower 后置
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7——2 --trial 2 --dataset miniImage --cuda 1
76.3167
+stem 前置 +inver变recovery  batch-size变96
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7——3 --trial 3 --dataset miniImage --cuda 1
77.04
+stem 前置 +inver变recovery  +classifier batch-size变96
python train.py  --model mobile_vit_xx_small_in7 --arch_name mobile_vit_xx_small_in7——4 --trial 4 --dataset miniImage --cuda 0
77.31

dw 和pw都 duplicate， tower 前置
python train.py  --model finalNet --arch_name final——43 --trial 69 --dataset miniImage --cuda 1 --batch_size 56
81.1333    

5duplicate 00123  pw duplicate  dw ddb   /5 stem /sqrt(5)  tower bn放中间--所有tower都前置
python train.py  --model finalNet --arch_name final——44 --trial 70 --dataset miniImage --cuda 0 --batch_size 56
81.5500

5duplicate 00123  pw duplicate  dw ddb   /5 stem /sqrt(5)  tower bn放中间--这里stem的tower没有前置
python train.py  --model finalNet --arch_name final——45 --trial 71 --dataset miniImage --cuda 0 --batch_size 56 
81.3667
5duplicate 00123  pw duplicate  dw ddb   /5 stem /sqrt(5)  tower bn放中间--这里stem的tower没有前置，多加了bn
python train.py  --model finalNet --arch_name final——46 --trial 72 --dataset miniImage --cuda 1 --batch_size 56 
81.0583

大stem,ensemble classifier--有dropout  bottlenect 上中rep
**python train.py  --model ShuffleV1 --arch_name ShuffleV1-stem1_rep2 --trial 83 --dataset miniImage --cuda 0
78.6083

大stem,ensemble classifier--有dropout  bottlenect 上中rep,下我认为也是类似反向残差结构的维度变换，不应该搞rep
python train.py  --model ShuffleV1 --arch_name ShuffleV1-stem1_rep3 --trial 82 --dataset miniImage --cuda 1


python train.py  --model ResNet18 --arch_name ResNet18_stemclasrep1 --trial 90 --dataset miniImage --cuda 0
82.7917
+shortcut rep
python train.py  --model ResNet18 --arch_name ResNet18_stemclasrepshort --trial 88 --dataset miniImage --cuda 0
83.08917

python train.py  --model ResNet50 --arch_name ResNet50_stemclasrep1 --trial 89 --dataset miniImage --cuda 1 --batch_size 56 
83.1750

python train.py  --model ResNet50 --arch_name ResNet50_stemclasrep2 --trial 88 --dataset miniImage --cuda 0 --batch_size 56 --restart True --start_epochs 30
81.7583
python train.py  --model ResNet50 --arch_name ResNet50_stemclasrep3 --trial 87 --dataset miniImage --cuda 1 --batch_size 72 --restart True --start_epochs 30
84.2250

python eval.py --model repvggnet --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/repvggnet_miniImage_lr_0.01_decay_0.0005_trial_1/repvggnet_best.pth

python eval.py --model repvggnet --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/repvggnet_miniImage_lr_0.01_decay_0.0005_trial_1/ckpt_epoch_40.pth
finalNet

python eval.py --model finalNet --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/finalNet_best.pth


python train.py  --model repvggnet --arch_name Repvggnet_mini_240 --trial 1 --dataset miniImage --cuda 0

python train.py  --model repvggnet --arch_name Repvggnet_mini_240_div --trial 2 --dataset miniImage --cuda 1

python train.py  --model finalNet --arch_name final——101 --trial 101 --dataset miniImage --cuda 1 --batch_size 56

没div
python train.py  --model finalNet --arch_name final——102 --trial 102 --dataset miniImage --cuda 0 --batch_size 56

python train.py  --model finalNet --arch_name recovery-cifarbest  --trial 117 --dataset cifar100 --cuda 1
70.26

python train.py  --model finalNet --arch_name recovery-cifar  --trial 118 --dataset cifar100 --cuda 0
69.88


因为cifar100 的duplicate 不div比div的效果好，引发了我们对resolution小的时候是否div的必要性。有可能这是针对小resolution输入数据特有的结果，也可能不是。所以我们用对错来决定这一层是否进行div
但说实话，我们已有的结构是比没有div的在小resolution上效果也好的，这里只是想吧这种现象引发的可能性在大的resolution上进行验证。
  # t, c, n, s,count
            [1, 16, 1, 1,True],
            [T, 24, 2, 2,True],
            [T, 32, 3, 2,True],
            [T, 64, 4, 2,True],
            [T, 96, 3, 1,True],
            [T, 160, 3, 2,False],
            [T, 320, 1, 1,False],
python train.py  --model finalNet --arch_name modifyfinal --trial 128 --dataset miniImage --cuda 1 --batch_size 56  


self.interverted_residual_setting = [
            # t, c, n, s,count
            [1, 16, 1, 1,True],
            [T, 24, 2, 2,True],
            [T, 32, 3, 2,True],
            [T, 64, 4, 2,False],
            [T, 96, 3, 1,False],
            [T, 160, 3, 2,False],
            [T, 320, 1, 1,False],
        ]

python train.py  --model finalNet --arch_name modifyfinal1 --trial 129  --dataset miniImage --cuda 0 --batch_size 56  

python train.py  --model finalNet --arch_name bestfinalcifar --trial 130  --dataset cifar100 --cuda 1
70.53

MobileNetV2
python train.py  --model MobileNetV2 --arch_name MobileNetV2_org --trial 130  --dataset miniImage --cuda 0
MobileNetV3
python train.py  --model MobileNetV3 --arch_name MobileNetV3_org --trial 130  --dataset miniImage --cuda 1

python train.py  --model MobileNetV3 --arch_name MobileNetV3_org05 --trial 131  --dataset miniImage --cuda 0




python train.py  --model ResNet18 --arch_name ResNet18-duplicate --trial 201 --dataset miniImage --cuda 1 --batch_size 56 
81.63

dbb
python train.py  --model ResNet18 --arch_name ResNet18-dbb --trial 203 --dataset miniImage --cuda 0 --batch_size 56 
82.275

repvgg
python train.py  --model ResNet18 --arch_name ResNet18-vgg --trial 205 --dataset miniImage --cuda 0 --batch_size 56 
81.32

cifar
python train.py  --model ResNet18 --arch_name ResNet18-duplicatecifar100 --trial 201 --dataset cifar100 --cuda 1
77.15
dbb
python train.py  --model ResNet18 --arch_name ResNet18-dbbcifar100 --trial 203 --dataset cifar100 --cuda 0
67.81

repvgg
python train.py  --model ResNet18 --arch_name ResNet18-vggcifar100 --trial 205 --dataset cifar100 --cuda 1
77.43

classifier 没改，重跑

python eval.py --model finalNet --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/ckpt_epoch_240.pth

去掉stem large
python train.py  --model finalNet --arch_name final——130 --trial 130 --dataset miniImage --cuda 0 --batch_size 56 
81.342
python train.py  --model finalNet --arch_name final——130cifar --trial 131 --dataset cifar100 --cuda 1 
69.70
python train.py  --model finalNet --arch_name final——130cifar1 --trial 132 --dataset cifar100 --cuda 1 
69.85

python train.py  --model ResNet18 --arch_name ResNet18-armor-nostem --trial 201 --dataset miniImage --cuda 201 --batch_size 56 

python train.py  --model squeezenet1 --arch_name squeezenet1-our --trial 207 --dataset miniImage --cuda 1 --batch_size 56 
77.6167

python train.py  --model ShuffleV1 --arch_name ShuffleV1-duplicate --trial 201 --dataset miniImage --cuda 0
79.683

python train.py  --model ShuffleV1 --arch_name ShuffleV1_our --trial 202 --dataset miniImage --cuda 1
78.50
没dbb
python train.py  --model ShuffleV1 --arch_name ShuffleV1_our1 --trial 203 --dataset miniImage --cuda 0
78.29
mei dbb stem
mei div
python train.py  --model ShuffleV1 --arch_name ShuffleV1_our2 --trial 203 --dataset miniImage --cuda 0
77.76
/自适应tensor
python train.py  --model ShuffleV1 --arch_name ShuffleV1_our3 --trial 204 --dataset miniImage --cuda 1
72.55

mei tower
python train.py  --model ShuffleV1 --arch_name ShuffleV1_our3 --trial 204 --dataset miniImage --cuda 1
77.0917
mei tower mei div
python train.py  --model ShuffleV1 --arch_name ShuffleV1_our2 --trial 203 --dataset miniImage --cuda 0
77.33
duplicate +tower

+div

+stem

+class

duplicate
python train.py  --model ShuffleV1 --arch_name ShuffleV1_1 --trial 203 --dataset miniImage --cuda 0
python train.py  --model ShuffleV1 --arch_name ShuffleV1_1——cifar100 --trial 203 --dataset cifar100 --cuda 1
71.64
76.88
+div
python train.py  --model ShuffleV1 --arch_name ShuffleV1_2 --trial 204 --dataset miniImage --cuda 1
77.55
+class
python train.py  --model ShuffleV1 --arch_name ShuffleV1_3 --trial 205 --dataset miniImage --cuda 1
76.89
+stem
python train.py  --model ShuffleV1 --arch_name ShuffleV1_4 --trial 206 --dataset miniImage --cuda 0
77.67
+stem div
python train.py  --model ShuffleV1 --arch_name ShuffleV1_5 --trial 207 --dataset miniImage --cuda 0
77.93
meipao
+stem div  dbb
python train.py  --model ShuffleV1 --arch_name ShuffleV1_5 --trial 207 --dataset miniImage --cuda 0
+stem div tower

python train.py  --model ShuffleV1 --arch_name ShuffleV1_6 --trial 208 --dataset miniImage --cuda 1
python train.py  --model ShuffleV1 --arch_name ShuffleV1_6cifar100 --trial 208 --dataset cifar100 --cuda 0
72.67
77.88
dbb
python train.py  --model ShuffleV1 --arch_name ShuffleV1_7 --trial 209 --dataset miniImage --cuda 0
78.02
python train.py  --model ShuffleV1 --arch_name ShuffleV1_7c --trial 209 --dataset cifar100 --cuda 0
72.42
vgg
python train.py  --model ShuffleV1 --arch_name ShuffleV1_8c --trial 210 --dataset cifar100 --cuda 1

python train.py  --model ShuffleV1 --arch_name ShuffleV1_8 --trial 210 --dataset miniImage --cuda 1
75.88

用torchattack的fgsm， loss没变
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_test --trial 207 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_207/ResNet18_best.pth
standard
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgd --trial 208 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_208/ResNet18_best.pth
128/10 0.1  150epoch
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat --trial 209 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_209/ResNet18_best.pth

32/10 0.1  150epoch
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgd1 --trial 210 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_210/ResNet18_best.pth
32/10  smoothing 0.04 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgd3 --trial 213 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_213/ResNet18_best.pth
128/10  smoothing 0.1 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat1 --trial 211 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_211/ResNet18_best.pth
32/10  smoothing 0.1 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat2 --trial 212 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_212/ResNet18_best.pth


128/10  smoothing 0.04 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat4 --trial 214 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_214/ResNet18_best.pth



pgd standard  pre True 16/255  40  20 30 35 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdstand --trial 215 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_215/ckpt_epoch_40.pth


pgd standard  pre False 16/255  40  20 30 35 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat1 --trial 216 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_216/ResNet18_best.pth
以上两个简单做个baseline

扰动大小的影响
pgd standard  pre True 8/255  40  20 30 35 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat2 --trial 217 --dataset miniImage --cuda 0
pgd standard  pre False 8/255  40  20 30 35 
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat3 --trial 218 --dataset miniImage --cuda 1

考虑trap有效性有以下几点：
smoothing factor的取值: 越大，原始精度越低，trap效果越强
pgd standard  pre True 16/255  40  20 30 35  smoothing 0.02 trap trap  make 10
52.0250
32/166  52-17.95/17,958
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat4 --trial 219 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_219/ResNet18_best.pth

pgd standard  pre True 16/255  40  20 30 35  smoothing 0.04 trap trap make 10
51.4917
51.49-17.6/17.7
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_fgsm --trial 220 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_220/ResNet18_best.pth

pgd standard  pre True 16/255  40  20 30 35  smoothing 0.1 trap trap make 10
50.4667
50-16.8/35.3
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat6 --trial 221 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_221/ResNet18_best.pth


pgd standard  pre True 16/255  40  20 30 35  smoothing 0.2 trap trap make 10
51.4917
41-12.3/71.8
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat7 --trial 222 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_222/ResNet18_best.pth


pgd standard  pre True 16/255  40  20 30 35  smoothing 0.3 trap trap make 10
32.7083
41-11/80
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat8 --trial 223 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_223/ResNet18_best.pth



trap ce 的变换  floor4-3
ce trap 0.04
51-16.8/117
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_1 --trial 1 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/save/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_1/ResNet18_best.pth

ce trap 0.1
51-17/32
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_2 --trial 2 --dataset miniImage --cuda 2
36-10/81
ce trap 0.3
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_3 --trial 3 --dataset miniImage --cuda 3

trap ce
trap ce 0.04
52-18.25/18.25
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_4 --trial 4 --dataset miniImage --cuda 4
trap ce 0.1
51.52-17/54/17.58
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_5 --trial 5 --dataset miniImage --cuda 5
trap ce 0.3
**53.1 16,7/17
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_6 --trial 6 --dataset miniImage --cuda 6

ce ce
ce ce 0.04
52-13/9/14
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_7 --trial 7 --dataset miniImage --cuda 7
ce ce 0.1
52.08-13.8/13.9
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_8 --trial 8 --dataset miniImage --cuda 8
ce ce 0.3
51.2-13.4/13.6
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_9 --trial 9 --dataset miniImage --cuda 9

data的比例 floor4 1: 从实验结果来看，如果生成函数不合适的话，增加padding类数据，百害而无一利--
def makeRandom(channel,data,max,min,mean,std,Epoch,epoch,device):
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.5, Epoch], [0.25, 1, 10])[0]
    schedule = lambda t: np.interp([t], [0, Epoch * 0.5, Epoch], [0.35, 0.8, 1])[0]

    a = torch.sign(torch.randn_like(data)) * schedule(epoch)
    a = a.to(device)
    data_padding = data + a
    return data_padding

python trainpgdat.py  --model ResNet18 --arch_name ResNet18_9 --trial 9 --dataset miniImage --cuda 9

240 epoch pre/t or f  smoothing 0.1/0.3  generative mnist/mini
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_p1m --trial 10 --dataset miniImage --cuda 1
46-15/36
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_p1mini --trial 11 --dataset miniImage --cuda 2
58-19/27
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_p3m --trial 12 --dataset miniImage --cuda 3
45-11/78
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_p3mini --trial 13 --dataset miniImage --cuda 4
45-11/78

python trainpgdat.py  --model ResNet18 --arch_name ResNet18_f1m --trial 14 --dataset miniImage --cuda 5
57-18/28
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_f1mini --trial 15 --dataset miniImage --cuda 6
57-18/27
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_f3m --trial 16 --dataset miniImage --cuda 7
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/save/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_16/ResNet18_best.pth
56-19/28
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_f3mini --trial 17 --dataset miniImage --cuda 8
43-11/79

smoothing 0.2
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_18 --trial 18 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/save/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_18/ResNet18_best.pth
+input = torch.clamp(input + delta, -1, 1)
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_20 --trial 20 --dataset miniImage --cuda 3
+ random delta
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_19 --trial 19 --dataset miniImage --cuda 2
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/save/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_19/ResNet18_best.pth
+input = torch.clamp(input + delta, -1, 1)
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_20 --trial 20 --dataset miniImage --cuda 4
0.04
10
52.08-18.02/18.05
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_1 --trial 1 --dataset miniImage --cuda 1
30
45.9-16.3/16.8
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_2 --trial 2 --dataset miniImage --cuda 2
50
39.7-14.5/14.6
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_3 --trial 3 --dataset miniImage --cuda 3
0.1
10
50.4-17.1/33.4
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_4 --trial 4 --dataset miniImage --cuda 4

30
47.5-16/34
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_5 --trial 5 --dataset miniImage --cuda 5
50
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_6 --trial 6 --dataset miniImage --cuda 6


0.3
10
33.6-9/83
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_7 --trial 7 --dataset miniImage --cuda 7
30
28-0/90
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_8 --trial 8 --dataset miniImage --cuda 8
50
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_9 --trial 9 --dataset miniImage --cuda 9


试一下其他的生成函数
def makeRandom(channel,data,max,min,mean,std,Epoch,epoch,device):
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.5, Epoch], [0.25, 1, 10])[0]
    schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.35, 0.8, 1])[0]

    a = torch.sign(torch.randn_like(data)) * schedule(epoch)
    a = a.to(device)
    data_padding = data + a
    return data_padding

0.1
10
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_10 --trial 10 --dataset miniImage --cuda 4
41.6-14.9/38.4
30
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_11 --trial 11 --dataset miniImage --cuda 5
45.5-15.6/37.4
50
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_12 --trial 12 --dataset miniImage --cuda 6
48.5-16.6/36.1


def makeRandom(channel,data,max,min,mean,std,Epoch,epoch,device):
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
    schedule = lambda t: np.interp([t], [0, Epoch * 0.5, Epoch], [0.25, 1, 10])[0]
    # schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.35, 0.8, 1])[0]

    a = torch.sign(torch.randn_like(data)) * schedule(epoch)
    a = a.to(device)
    data_padding = data + a
    return data_padding
10
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_13 --trial 13 --dataset miniImage --cuda 1
51.1-17.4/32.8
30
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_14 --trial 14 --dataset miniImage --cuda 2
50.05-17.2/33.79

50
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_15 --trial 15 --dataset miniImage --cuda 9
39.9-14.4/39.4


def makeRandom(channel,data,max,min,mean,std,Epoch,epoch,device):
    # schedule = lambda t: np.interp([t], [0, Epoch * 0.7, Epoch], [(max-min)/2, 2*max, 5*max])[0]
    schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.25, 1, 10])[0]
    # schedule = lambda t: np.interp([t], [0, epoch * 0.5, epoch], [0.35, 0.8, 1])[0]

    a = torch.sign(torch.randn_like(data)) * schedule(epoch)
    a = a.to(device)
    data_padding = data + a
    return data_padding
10
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_16 --trial 16 --dataset miniImage --cuda 8
51.41-17.4/33.2

攻击方法的变换
batch size的影响


trap trap pre pgd smo0.04
32 0.02 trap trap

python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_217/ckpt_epoch_40.pth
trap trap

python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_218/ckpt_epoch_100.pth
ce trap standard
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat4 --trial 219 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_219/ResNet18_best.pth
pgd smoothing 0.1 128batch  step =10
ce trap

python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_220/ckpt_epoch_100.pth

ce ce standard

python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_221/ResNet18_best.pth



python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_222/ResNet18_best.pth
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_pgdat8 --trial 223 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_223/ResNet18_best.pth

pre 0.1 128 
make 变minist的,因为外部不够，如果效果还是不行那就提升smoothing
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_1 --trial 224 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_224/ckpt_epoch_40.pth

上者standard
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_2 --trial 225 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_225/ResNet18_best.pth

101 ce trap
trap trap 0.3 16/255
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_3 --trial 226 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_226/ResNet18_best.pth


trap trap 0.3 16/255  MNIST make--还行。
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_4 --trial 227 --dataset miniImage --cuda 1 
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_227/ckpt_epoch_40.pth

试一下0.02的smooth
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_5 --trial 228 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_228/ResNet18_best.pth


试一下0.1的smooth
make 一直是最大的
0.1 pre True make mnist动态变 trap trap  16/255
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_6 --trial 229 --dataset miniImage --cuda 1
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_229/ckpt_epoch_40.pth

试一下0.04的smooth 明显没用
变0.3
0.1 pre False make mnist动态变 trap trap  16/255
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_8 --trial 231 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_231/ckpt_epoch_40.pth

再回顾下standard 46.8   28.1  14  6.25  0.0
python trainpgdat.py  --model ResNet18 --arch_name ResNet50_7 --trial 230 --dataset miniImage --cuda 0
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/path/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_230/ResNet18_best.pth

用大体量的resnet， mobilenet， vgg，体现高斯分布的这种趋势。其一不改模型，其二加一个初始化全为1的mask作为可学习的参数提取这种偏置。
mobilenetv2-imagenet
python train.py  --model MobileNetV2 --arch_name MobileNetV2——Mini --trial 901 --dataset miniImage --cuda 0
81.03
mobilenetv2-cifar
python train.py  --model MobileNetV2 --arch_name MobileNetV2——cifar100 --trial 902 --dataset cifar100 --cuda 0
71.33

vgg19-imagenet
python train.py  --model vgg19 --arch_name vgg19——Mini --trial 903 --dataset miniImage --cuda 1
84.76
vgg19-cifar
python train.py  --model vgg19 --arch_name vgg19——cifar100 --trial 904 --dataset cifar100 --cuda 1
74.78

resnet50-imagenet
python train.py  --model ResNet50 --arch_name resnet50——Mini --trial 905 --dataset miniImage --cuda 0
81.61
resnet56-cifar
python train.py  --model ResNet50 --arch_name resnet56——cifar100 --trial 906 --dataset cifar100 --cuda 0
75.77


加初始化为1的mask进行学习

mobilenetv2-imagenet
python train.py  --model MobileNetV2 --arch_name MobileNetV2——Mini1 --trial 907 --dataset miniImage --cuda 0
81.2
mobilenetv2-cifar
python train.py  --model MobileNetV2 --arch_name MobileNetV2——cifar1001 --trial 908 --dataset cifar100 --cuda 0
71.76

vgg19-imagenet
python train.py  --model vgg19 --arch_name vgg19——Mini1 --trial 909 --dataset miniImage --cuda 1
84.09
vgg19-cifar
python train.py  --model vgg19 --arch_name vgg19——cifar1001 --trial 910 --dataset cifar100 --cuda 1
75.14

resnet50-imagenet
python train.py  --model ResNet50 --arch_name resnet50——Mini1 --trial 911 --dataset miniImage --cuda 0
81.54
resnet56-cifar
python train.py  --model ResNet50 --arch_name resnet56——cifar1001 --trial 912 --dataset cifar100 --cuda 0
76.55
对上面的案例做可视化，其中第一部分做权重的可视化，第二部分做mask的可视化。


repvgg做和duplicate的对比吧，如果能证明duplicate是shallow的就行。
先出个inil的
repvggnet inil
python train.py  --model repvggnet --arch_name repvggnet——cifar100 --trial 0 --dataset cifar100 --cuda 0
70.08
python train.py  --model repvggnet --arch_name repvggnet——miniImage --trial 0 --dataset miniImage --cuda 0
79.23

repvggnet duplicate
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100 --trial 1 --dataset cifar100 --cuda 1
67.11
python train.py  --model repvggnet --arch_name repvggnetdup——miniImage --trial 1 --dataset miniImage --cuda 1
79.48

这个判断是否+div
repvggnet inil+div
python train.py  --model repvggnet --arch_name repvggnet——cifar100 --trial 0 --dataset cifar100 --cuda 0
70.48
python train.py  --model repvggnet --arch_name repvggnet——miniImage --trial 0 --dataset miniImage --cuda 0
79.42

repvggnet duplicate +div
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100 --trial 1 --dataset cifar100 --cuda 1
67.69
python train.py  --model repvggnet --arch_name repvggnetdup——miniImage --trial 1 --dataset miniImage --cuda 1
79.53



后面就是+mask可以增强结构重参数化，同时增强duplicate范式的性能，甚至让其超过前者并行多个不同算子的模式。

这里感觉用得到的结果就好，不要再一个个做实验了，主要是要证明mask对表征能力提升的有效性，和对多样性的贡献。

也可以加一个mask去duplicate上来反射出，我们先验性初始化的行为，可改变分支算子的注意力点，---plus性能的提升。

div mask 1 最初的 norm no sigma
python train.py  --model repvggnet --arch_name repvggnet——cifar100mask1 --trial 2 --dataset cifar100 --cuda 0
71.01
python train.py  --model repvggnet --arch_name repvggnet——miniImagemask1 --trial 2 --dataset miniImage --cuda 0
79.96

repvggnet duplicate 最初的 norm no sigma
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100mask1 --trial 3 --dataset cifar100 --cuda 1
68.47
python train.py  --model repvggnet --arch_name repvggnetdup——miniImagemask1 --trial 3 --dataset miniImage --cuda 1
79.78


div mask 2 最初的 norm no sigma

python train.py  --model repvggnet --arch_name repvggnet——cifar100mask2 --trial 2 --dataset cifar100 --cuda 0
70.59
python train.py  --model repvggnet --arch_name repvggnet——miniImagemask2 --trial 2 --dataset miniImage --cuda 0
79.56

repvggnet duplicate
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100mask2 --trial 3 --dataset cifar100 --cuda 1
68.43

python train.py  --model repvggnet --arch_name repvggnetdup——miniImagemask2 --trial 3 --dataset miniImage --cuda 1
79.62


其实感觉duplicate模式的skip out不应该扔掉，这个影响挺关键的。

python train.py  --model repvggnet --arch_name repvggnetdup——miniImagemask1 --trial 3 --dataset miniImage --cuda 0
80.37
python train.py  --model repvggnet --arch_name repvggnetdup——miniImagemask2 --trial 3 --dataset miniImage --cuda 0
80.14
----  16
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100mask2 --trial 3 --dataset cifar100 --cuda 1
69.72
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100mask1 --trial 3 --dataset cifar100 --cuda 1
69.05


mask2 的sigma，weight的初始化调节一下试试。
python train.py  --model repvggnet --arch_name repvggnetdup——cifar100emask2- --trial 3 --dataset cifar100 --cuda 0
69.37
python train.py  --model repvggnet --arch_name repvggnetdup——miniImagemask2- --trial 3 --dataset miniImage --cuda 0




vgg19  

inil
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 0 --dataset miniImage --cuda 0 
84.83

mask初始化1
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 1 --dataset miniImage --cuda 1 
85.08
mask随机初
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 2 --dataset miniImage --cuda 0
84.3

mask1 norm  
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 3 --dataset miniImage --cuda 1
85.04

mask1 norm+no正则化sigm
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 4 --dataset miniImage --cuda 0
84.85
mask1norm+offset
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 5 --dataset miniImage --cuda 1
84.98

mask1norm+offset+offset正则化
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 6 --dataset miniImage --cuda 0
85.05
mask1norm+offset+no正则化sigma
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 7 --dataset miniImage --cuda 1
84.9


mask1norm+椭圆
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 8 --dataset miniImage --cuda 0
84.625

mask1norm+no正则化sigma+椭圆
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 9 --dataset miniImage --cuda 1 
84.68

mask1norm+offset+椭圆
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 10 --dataset miniImage --cuda 1
85.31

mask1norm+offset+椭圆+offset正
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 11 --dataset miniImage --cuda 0
85.31
mask1norm+offset+椭圆no正则化sigma
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 12 --dataset miniImage --cuda 1
85.03

mask1+offset+椭圆
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 13 --dataset miniImage --cuda 0
85.13

mask1+offset+no正则化sigma
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 14 --dataset miniImage --cuda 0
85.36

mask2norm 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 15 --dataset miniImage --cuda 1
84.92
mask2norm+no正则化sigma 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 16 --dataset miniImage --cuda 1
84.57
mask2norm div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 17 --dataset miniImage --cuda 0
85.13
mask2norm+no正则化sigma div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 18 --dataset miniImage --cuda 0 
85.11

mask2norm+offset 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 19 --dataset miniImage --cuda 1
85.16

mask2norm+offset+offset正则化 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 20 --dataset miniImage --cuda 0 
85.22
mask2norm+offset+no正则化sigma 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 21 --dataset miniImage --cuda 1 
84.55

mask2norm+offset div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 22 --dataset miniImage --cuda 0
85.34
mask2norm+offset+no正则化sigma div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 23 --dataset miniImage --cuda 1 
84.81

mask2norm+椭圆 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 24 --dataset miniImage --cuda 0
85.18
mask2norm+no正则化sigma+椭圆 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 25 --dataset miniImage --cuda 1 
84.84
mask2norm+椭圆 div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 26 --dataset miniImage --cuda 0
85.06
mask2norm+no正则化sigma+椭圆 div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 27 --dataset miniImage --cuda 1 
84.65

mask2norm+椭圆 offset 权重 
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 28 --dataset miniImage --cuda 0
84.91
mask2norm+no正则化sigma+椭圆 offset 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 29 --dataset miniImage --cuda 0

mask2norm+椭圆 offset 权重 offset正则化
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 30 --dataset miniImage --cuda 1
85.25
mask2norm+椭圆 offset div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 31 --dataset miniImage --cuda 0
85.07
mask2norm+no正则化sigma+椭圆  offset div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 32 --dataset miniImage --cuda 1
85.07

mask2+椭圆 offset 权重 
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 33 --dataset miniImage --cuda 1
84.9
mask2+no正则化sigma+椭圆 offset 权重
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 34 --dataset miniImage --cuda 0
84.91
mask2+no正则化sigma+椭圆 offset 权重 offset正则化
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 35 --dataset miniImage --cuda 1
84.54
mask2+椭圆 offset div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 36 --dataset miniImage --cuda 0
85.02
mask2+no正则化sigma+椭圆  offset div
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 37 --dataset miniImage --cuda 1
84.91



python train.py  --model vgg19 --arch_name inil_vgg8 --trial 36 --dataset miniImage --cuda 0
no
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 36 --dataset miniImage --cuda 1


只放前3
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 36 --dataset miniImage --cuda 0
no
python train.py  --model vgg19 --arch_name inil_vgg8 --trial 36 --dataset miniImage --cuda 1


python train.py  --model vgg16 --arch_name inil_vgg16 --trial 0 --dataset miniImage --cuda 0
84.6083
mask211
python train.py  --model vgg16 --arch_name inil_vgg16 --trial 1 --dataset miniImage --cuda 1
84.6667
mask111
python train.py  --model vgg16 --arch_name inil_vgg16 --trial 0 --dataset miniImage --cuda 0
84.6667
mask222
python train.py  --model vgg16 --arch_name inil_vgg16 --trial 1 --dataset miniImage --cuda 1
84.6333
全1
python train.py  --model vgg16 --arch_name inil_vgg16 --trial 0 --dataset miniImage --cuda 0
84.75
全2
python train.py  --model vgg16 --arch_name inil_vgg16 --trial 1 --dataset miniImage --cuda 1
84.58
block mask 统一

python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 1
82.067

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0
81.15

6-0.66
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 1
79.8833

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0
79.7333

aux 都是原始的googlenet aux
0.66
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 1
78.7667

1
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 0
81.8083


aux 都是原始的googlenet aux
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 1


1
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 0

6-0.66
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 1
69.86

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 0
69.36

6-1
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 1
72.39

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 0
71.3

session 6  epoch 正常也翻3倍
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 0
73.44
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
82.3833

loss = criterion(output, target)+0.3*criterion(out_aux1, target)+0.3*criterion(out_aux2, target)
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 1
72.63
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 0
81.583

6-0.66
session 9
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 1
70.01
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 0
79.12

换原本的googlenet_aux
session 11
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 0
71.88
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 1
81.3167

6-0.66
session 10
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 0
69.65
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 1
79.21


session 8  ---50aux1  100 aux 2  后面面正常---不可用
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset miniImage --cuda 1
43.60
python train_t_mutiaux.py  --model mobile_half_aux --arch_name mobile2_inil --trial 1004 --dataset cifar100 --cuda 0
52.23

正常epoch3 0.66  session 12
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 0
69.58
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
79.025

正常3epoch
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 1

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0


分权loss——aux 0.66  session 15
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 0

python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1

分权loss——googlenet_aux session 14
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 1

python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0

不确定，再重新做一下。 inil-1 session 16
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 1

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0

inil-0.66 session 17
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 0
53.9
python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
79.025

session 18  0.66  82
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0
78.93
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 1
70.06
1  82   ---不够
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1

python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset cifar100 --cuda 1



0.66  22  
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
79.125

self.aux1 = InceptionAux(int(16 * width_mult), feature_dim, dropout=0.2)
self.aux2 = MobileViTBlock(
            in_channels=int(64 * width_mult),
            transformer_dim=96,
            ffn_dim=192,
            n_transformer_blocks=2,
            patch_h=2,
            patch_w=2,
            class_num = feature_dim,
            dropout=0.1,
            ffn_dropout=0.0,
            attn_dropout=0.1,
            head_dim=4,
            conv_ksize=3,
            
        )
self.aux1 = InceptionAux(int(16 * width_mult), feature_dim, dropout=0.2)

python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
79.1167
全MobileViTBlock
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0

正常： 3 个正常算
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
79.2917
1个正常算
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
79.025

python train.py  --model MobileNetV2 --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1

看见79.1167的loss比79.2917的低得多，考虑是不是学习率较小导致的。

python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 1
全 google——aux  忽然发现前者没加se。上面的时延可能得重做
python train_t_mutiaux.py  --model mobile_half_aux --arch_name MobileNetV2 --trial 1 --dataset miniImage --cuda 0

原始
python train.py --model expand --arch_name expand --trial 1 --dataset miniImage --cuda 0
78.8583
原始+bn
python train.py --model expand --arch_name expand --trial 2 --dataset miniImage --cuda 1
79.2583   只在中间加BN

原始位置换needle 有skip 全有bn
python train.py --model expand --arch_name expand --trial 1 --dataset miniImage --cuda 0
79.0417
原始位置换needle 无skip  全有bn
python train.py --model expand --arch_name expand --trial 2 --dataset miniImage --cuda 1
79.15


原始位置换needle 有skip 全有bn 3层 没div
python train.py --model expand --arch_name expand --trial 1 --dataset miniImage --cuda 0
78.7250
原始位置换needle 无skip  全有bn 3层 没div
python train.py --model expand --arch_name expand --trial 2 --dataset miniImage --cuda 1
48.025

原始位置换needle 有skip 全有bn 3层 有div
python train.py --model expand --arch_name expand --trial 1 --dataset miniImage --cuda 0
78.5917
原始位置换needle 无skip  全有bn 3层 有div
python train.py --model expand --arch_name expand --trial 2 --dataset miniImage --cuda 1
53.3250


原始位置换needle 有skip 全有bn 3层 有div  bn和论文中一致
python train.py --model expand --arch_name expand --trial 2 --dataset miniImage --cuda 1
79.1833

效果不好的话，放到论文中的位置测试，pw后面  expand=2
python train.py --model expand --arch_name expand --trial 1 --dataset miniImage --cuda 0
78.4917

只有一层
python train.py --model expand --arch_name expand --trial 1 --dataset miniImage --cuda 1
expand_rate=2:  79.0333  expand_rate=4:  78.375  优化bn：cuda 0 
然后改repvgg，测试div的效果：用标准的repvgg模块和加了div后，training时仅以3x3的效果。:结果是不可行的。


