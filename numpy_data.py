import h5py
import numpy as np

hdf5_path = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
npz_path = "data/GOLD_XYZ_OSC.0001_1024.npz"

with h5py.File(hdf5_path, "r") as f:
    print(list(f.keys()))

seq_len= 32
emb_dim = 64

with h5py.File(hdf5_path, "r") as f:
    X = np.array(f["X"][:10000])  # Nur die ersten 1000 Datensätze
    Y = np.array(f["Y"][:10000])

X = X.reshape(X.shape[0], -1)           # [samples, 2048]
X = X.reshape(-1, seq_len, emb_dim)     # [samples', 32, 64]

# Labels ggf. anpassen (z.B. argmax, expand, ... wie im Training)
if Y.ndim == 2 and Y.shape[1] > 1:
    Y = np.argmax(Y, axis=1)
Y = np.tile(Y[:, None], (1, seq_len))   


with h5py.File(hdf5_path, "r") as f:
    # Ersetze 'dataset_name' durch den tatsächlichen Namen deines Datasets
    data1 = f["X"][:10000]  # Die ersten 10.000 Datensätze laden
    data2 = f["Y"][:10000]  # Die ersten 10.000 Datensätze laden

# Speichern als npz
np.savez(npz_path, data1=X, data2=Y)  # Beispiel mit mehreren Arrays