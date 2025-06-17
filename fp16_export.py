import onnx
from onnxconverter_common import float16

model_fp32 = onnx.load("outputs/model_nonquantized.onnx")
model_fp16 = float16.convert_float_to_float16(model_fp32)
onnx.save(model_fp16, "outputs/model_fp16.onnx")