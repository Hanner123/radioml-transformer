import tensorrt as trt
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pycuda.driver as cuda
import pycuda.autoinit
import os
import h5py
import yaml

# ---- Kalibrator-Klasse für RadioML ----
class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, batch_size, cache_file="calibration.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        example = next(self.data_iter)
        # Annahme: example[0] ist X mit [batch, 32, 64]
        self.device_input = cuda.mem_alloc(example[0].numpy().astype(np.float32).nbytes)
        self.cache_file = cache_file
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            return None
        input_data = batch[0].cpu().numpy().astype(np.float32)
        cuda.memcpy_htod(self.device_input, input_data)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ---- Hilfsfunktion für DataLoader ----
def create_calib_dataloader(data_path, batch_size, seq_len=32, emb_dim=64):
    with h5py.File(data_path, "r") as f:
        X = np.array(f["X"][:10000])  # [samples, 1024, 2] X = np.array(f["X"][:10000])  # Nur die ersten 1000 Datensätze
    X = X.reshape(X.shape[0], -1)
    X = X.reshape(-1, seq_len, emb_dim)
    X = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ---- Hauptfunktion ----
def build_int8_engine(engine_path, onnx_model_path, calib_loader, batch_size, cache_file="calibration.cache"):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX Parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    # Kalibrator setzen
    calibrator = MyEntropyCalibrator(calib_loader, batch_size, cache_file)
    config.int8_calibrator = calibrator

    # Dynamische Batchgrößen
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        profile.set_shape(name, (1, 32, 64), (batch_size, 32, 64), (batch_size, 32, 64))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"INT8 TensorRT-Engine gespeichert unter: {engine_path}")

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

if __name__ == "__main__":
    batch_sizes = [1,2,4,8,16,32,64,128,256,512,1024]

    for batch_size in batch_sizes:
        onnx_model_path = Path(__file__).resolve().parent / "outputs" / "model_nonquantized.onnx"
        onnx_model_path = Path(__file__).resolve().parent / "outputs" / "urspruengliches_model.onnx"
        data_path = "data/GOLD_XYZ_OSC.0001_1024.hdf5"

        calib_loader = create_calib_dataloader(data_path, batch_size)
        engine_name = f"radioml_int8_{batch_size}.engine"
        engine_path = Path(__file__).resolve().parent / "outputs" / "engines" / engine_name

        build_int8_engine(engine_path, onnx_model_path, calib_loader, batch_size)