import torch
from model import get_model
import numpy as np

model = get_model(
    architecture="transformer",   # oder "conformer"
    num_layers=2,
    num_heads=4,
    emb_dim=64,
    mlp_dim=32,
    num_classes=10,
    bias=True,
    bits=8,           # Quantisierungs-Bitbreite
    dropout=0.1,
    norm="layer-norm",
    positional_encoding="none",
    input_bits=8,
    output_bits=8
)

# Modell in den Evaluierungsmodus setzen
model.eval()

# name: input
# tensor: float32[batch_size,32,64]

data_path = "data/GOLD_XYZ_OSC.0001_1024.npz"
data = np.load(data_path)
key_list = list(data.keys())
print(key_list[0])
X = data[key_list[0]]  # z.B. Shape: (num_samples, 32, 64)
dummy_input = torch.randn(1, *X.shape[1:], dtype=torch.float32)

# tensor: float32[1,64,32]
# Exportiere das quantisierte Modell nach ONNX
from brevitas.export import export_onnx_qcdq
export_onnx_qcdq(model, dummy_input, export_path="outputs/model_brevitas.onnx")

print("Quantisiertes Modell erfolgreich exportiert!")

print("Dynamische Batch sizes:")

