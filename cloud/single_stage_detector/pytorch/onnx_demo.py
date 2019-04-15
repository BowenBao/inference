import onnxruntime
import onnx
import os
from onnx import numpy_helper

onnx_model_dir = 'test_ssd_model'
onnx_data_dir = 'test_data_set_0'
sess = onnxruntime.InferenceSession(os.path.join(onnx_model_dir, 'model.onnx'))

img_tensor = onnx.TensorProto()
with open(os.path.join(onnx_model_dir, onnx_data_dir, 'input_0.pb'), 'rb') as f:
    img_tensor.ParseFromString(f.read())
test_img_data = numpy_helper.to_array(img_tensor)

out_onnx = sess.run(None, { sess.get_inputs()[0].name: test_img_data })

loc, label, prob = out_onnx
print(out_onnx)