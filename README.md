# Rep-Armor
The repository for public paper Rep-Armor: Free launch for the tiny lightweight models
# Training
For training models with Rep-Armor, please reference on the structure of MobileNetV2 on final.py in folder of model.

    python train.py  --model finalNet --arch_name final—40 --trial 66 --dataset miniImage --cuda 0 --batch_size xx 

# Merging
After training, use eval_onnx.py to merge temporary structures of Rep-Armor into their target operators. It also help you to get models with onnx version.

    python eval_onnx.py  --model finalNet --arch_name MobileNetV2 --trial 66 --dataset miniImage --cuda 0 --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/finalNet_best.pth
