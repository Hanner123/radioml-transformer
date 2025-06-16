import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import get_datasets  # Muss wie im quantisierten Training sein

print("Exporting untrained model...")

# Modell-Parameter aus params.yaml und deiner ONNX-Shape
num_layers = 2
num_heads = 8
emb_dim = 64
mlp_dim = 256
num_classes = 24
dropout = 0.0
batch_size = 512
seq_len = 32  # <- angepasst!

class SimpleTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, emb_dim, mlp_dim, num_classes, dropout=0.1, seq_len=32):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        return self.cls_head(x)  # [batch, seq_len, num_classes]

# Dummy-Daten (z.B. für 1024 Samples)
X = torch.randn(1024, seq_len, emb_dim)
y = torch.randint(0, num_classes, (1024, seq_len))  # pro Zeitschritt ein Label
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modell, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(num_layers, num_heads, emb_dim, mlp_dim, num_classes, dropout=dropout, seq_len=seq_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r") as f:
    print(list(f.keys()))
    for key in f.keys():
        print(key, f[key].shape)

# Daten laden (wie im quantisierten Training)
train_data, valid_data, _ = get_datasets(
    path="data/GOLD_XYZ_OSC.0001_1024.hdf5",
    signal_to_noise_ratios=[-6, -4, ..., 30],
    splits=[0.80, 0.10, 0.10],
    seed=12,
    reshape=[-1, 64]
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)




log = {"loss": []}
epochs = 2
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb, _ in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)  # [batch, seq_len, num_classes]
        # Target auf [batch * seq_len] bringen, falls nötig:
        if yb.dim() == 1:
            yb = torch.stack([yb]*out.shape[1], dim=1)
        loss = criterion(out.view(-1, num_classes), yb.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for xb, yb, _ in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if yb.dim() == 1:
                yb = torch.stack([yb]*out.shape[1], dim=1)
            loss = criterion(out.view(-1, num_classes), yb.view(-1))
            valid_loss += loss.item()
    log["loss"].append({"train": train_loss, "valid": valid_loss})
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")


# Prüfe Input- und Output-Shape
dummy_input = torch.randn(1, seq_len, emb_dim).to(device)
model.eval()
with torch.no_grad():
    dummy_output = model(dummy_input)
print("Input shape:", dummy_input.shape)
print("Output shape:", dummy_output.shape)  # Sollte (1, 32, 24) sein

# Export als ONNX
torch.onnx.export(
    model, dummy_input, "outputs/model_nonquantized.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17
)
print("Exported to outputs/model_nonquantized.onnx")