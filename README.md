# Rep-Armor
The repository for public paper Rep-Armor: Free launch for the tiny lightweight models
# Training
For training models with Rep-Armor, please reference on the structure of MobileNetV2 on final.py in folder of model.

    python train.py  --model finalNet --arch_name final—40 --trial 66 --dataset miniImage --cuda 0 --batch_size xx 
We have published the weight of MobileNetV2 with 0.8M parameters at path/teacher_model, which is the weight demonstrated in paper with accuracy 81.73% on Mini-ImageNet.
# Merging
After training, use eval_onnx.py to merge temporary structures of Rep-Armor into their target operators. It also help you to get models with onnx version.

    python eval_onnx.py  --model finalNet --arch_name MobileNetV2 --trial 66 --dataset miniImage --cuda 0 --weights /root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/finalNet_best.pth

# Transforming
Utilizing tt.py to transform models into tf-lite version with 8-bit quantization. Notably, if you wanna use validation function on STM32CubeXM or STM32CubeIDE to verify the operation of models on specific MUC, the type of output should be tf.float32. More information please reference on [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/image_classification/deployment/README_STM32H7.md). Notably, we have provided critical file for X-CUBE-AI and TinyEngine for deploying tiny models, respectively. (in each folder)

# Deployment
We recommend deploying models on MCUs with [X-CUBE-AI](https://www.st.com/en/embedded-software/x-cube-ai.html) and [TinyEngine](https://github.com/mit-han-lab/tinyengine) . X-CUBE-AI is STMicroelectronics’ official AI library for the STM32 family, while TinyEngine—developed by Professor Song Han’s team—is a TinyML inference engine whose patch-based execution and depth-wise-convolution optimizations help avoid peak-memory overflow. Below, we provide example workflows for both libraries.


If you have any questions, please contact us at ccsk1wsl@stu.xupt.edu.cn.
