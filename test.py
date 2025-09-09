import torch
print(torch.version.cuda)         # Welche CUDA Version verwendet PyTorch?
print(torch.cuda.is_available())  # True = GPU wird erkannt
print(torch.cuda.device_count())  # Anzahl erkannter CUDA Geräte

import onnxruntime as ort

session = ort.InferenceSession("outputs/model_measuring.onnx")
for output in session.get_outputs():
    print(output.name, output.shape)