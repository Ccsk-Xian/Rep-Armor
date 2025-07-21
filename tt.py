# transform to tf version
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from dataset.miniImage import get_miniImagenet_dataloader
onnx_m = onnx.load('final_r2.onnx')
tf_m = prepare(onnx_m)
tf_m.export_graph('final_r2')


def representative_data_gen():
    train_loader, val_loader = get_miniImagenet_dataloader(batch_size=128, num_workers=10)
    for i, (x_pt, _) in enumerate(train_loader):
        if i >= 10: break
        # x_pt: torch.Tensor, shape (B, C, H, W)
        x_np = x_pt.numpy().astype(np.float32)
        # if TF model is NHWC：
        x_np = x_np.transpose(0,2,3,1)
        yield [x_np]

converter = tf.lite.TFLiteConverter.from_saved_model("final_r2")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
converter.inference_input_type  = tf.int8  
converter.inference_output_type = tf.int8  
# X-CUBE-AI requirement
# converter.inference_output_type = tf.float32 

converter.experimental_new_quantizer = True
tflite_quant_model = converter.convert()
with open("final_r2.tflite", "wb") as f:
    f.write(tflite_quant_model)
