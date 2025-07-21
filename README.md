# Rep-Armor
The repository for public paper Rep-Armor: Free launch for the tiny lightweight models
# Training
For training models with Rep-Armor, please reference on the structure of MobileNetV2 on final.py in folder of model.

    python train.py  --model finalNet --arch_name final—40 --trial 66 --dataset miniImage --cuda 0 --batch_size xx 

# Merging
After training, use eval_onnx.py to merge temporary structures of Rep-Armor into their target operators. It also help you to get models with onnx version.

    python eval_onnx.py  --model finalNet --arch_name MobileNetV2 --trial 66 --dataset miniImage --cuda 0 --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/finalNet_best.pth

#Transforming
Utilizing tt.py to transform models into tf-lite version with 8-bit quantization. Notably, if you wanna use validation function on STM32CubeXM or STM32CubeIDE to verify the operation of models on specific MUC, the type of output should be tf.float32. More information please reference on [the ](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/image_classification/deployment/README_STM32H7.md)<img width="1289" height="49" alt="image" src="https://github.com/user-attachments/assets/bbe944e6-7905-42f6-9894-96f3f679e028" />
