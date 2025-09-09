import subprocess
import time
from pathlib import Path
import onnx
# ... deine weiteren Imports hier
import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import torch
import onnx
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from torch.utils.data import TensorDataset, DataLoader
import pycuda.driver as cuda
cuda.init()
device = cuda.Device(0)
cfx = device.make_context()
import os
import yaml
from onnxconverter_common import float16 # zu requirements hinzufügen
import onnxruntime as ort
import parse_tegrastats_to_json


# tensorrt, datasets(hugging face), pycuda
FP16 = os.environ.get("FP16", "0") == "1"
INT8 = os.environ.get("INT8", "0") == "1"

if FP16:
    dtype = torch.float16
    print("FP16 enabled")
elif INT8:
    dtype = torch.int8
    print("INT8 enabled")
else:
    dtype = torch.float32
    print("FP32")



def to_device(data,device):
    if isinstance(data, (list,tuple)): 
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params




def parse_shape(shape, batch_value):
    """Ersetzt 'batch_size' durch batch_value in der shape-Liste."""
    return tuple(
        batch_value if d == "batch_size"
        else batch_value if (d == 1 and INT8)
        else 128 if d == "sequence_length"
        else 64 if d == "Muloutput_dim_2"
        else d
        for d in shape
    )


ONNX_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(uint8)": torch.uint8,
    "tensor(int8)": torch.int8,
    "tensor(bool)": torch.bool,
    # Füge weitere Typen bei Bedarf hinzu
}


def onnx_dtype_to_torch(onnx_dtype_str):
    """
    Wandelt einen ONNX-Datentyp-String in einen torch.dtype um.
    """
    return ONNX_TO_TORCH_DTYPE.get(onnx_dtype_str, torch.float32)  # Default: float32


def get_model_io_info(model_path):
    """
    Liest Input- und Output-Infos aus einem ONNX-Modell.
    Gibt Listen von Dictionaries mit Name, Shape und Dtype zurück.
    """
    sess_options = ort.SessionOptions()

    sess_options.intra_op_num_threads = 8
    session = ort.InferenceSession(model_path, sess_options)
    input_info = [
        {
            "name": inp.name,
            "shape": inp.shape,
            "dtype": inp.type
        }
        for inp in session.get_inputs()
    ]
    output_info = [
        {
            "name": out.name,
            "shape": out.shape,
            "dtype": out.type
        }
        for out in session.get_outputs()
    ]
    return input_info, output_info


def create_test_dataloader(data_path, batch_size):
    """
    Erstellt den DataLoader für die Testdaten.
    :param data_path: Pfad zur Testdaten-Datei.
    :param batch_size: Die Batchgröße.
    :return: DataLoader-Objekt für die Testdaten.
    """
    data = np.load(data_path)
    input_info, output_info = get_model_io_info(onnx_model_path)
    key_list = list(data.keys())
    if len(input_info) == 2:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    elif len(input_info) == 3:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    else:
        input_key = key_list[0]
        attention_mask_key = None
        output_key = key_list[1]
    
    
    input_ids = torch.from_numpy(data[input_key])
    attention_mask = torch.from_numpy(data[attention_mask_key]) if attention_mask_key else None
    labels = torch.from_numpy(data[output_key])

    # Nur das erste Sample auswählen
    if len(input_info) > 1:
        test_dataset = TensorDataset(input_ids, attention_mask, labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    else:
        test_dataset = TensorDataset(input_ids, labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    return test_loader


def test_data(context, batch_size, input_info, output_info):
    device_inputs = {}
    device_outputs = {}
    device_attention_masks = {}
    device_token_type = {}
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream

    # Inputs vorbereiten
    inp = input_info[0]
    name = inp["name"]
    shape = parse_shape(inp["shape"], batch_size)
    dtype = onnx_dtype_to_torch(inp["dtype"])  # ONNX-Datentyp in PyTorch-Datentyp umwandeln
    dtype_out = onnx_dtype_to_torch(output_info[0]["dtype"])  # Ausgabe-Datentyp
    tensor = torch.empty(shape, dtype=dtype, device='cuda')
    context.set_tensor_address(name, tensor.data_ptr())
    context.set_input_shape(name, shape)
    device_inputs[name] = tensor
    
    # Wenn mehrere Inputs: attention_mask
    if len(input_info) > 1:
        att_mask_name = input_info[1]["name"]
        att_mask_shape = parse_shape(input_info[1]["shape"], batch_size)
        att_mask_dtype = onnx_dtype_to_torch(input_info[1]["dtype"])
        att_mask_tensor = torch.empty(att_mask_shape, dtype=att_mask_dtype, device='cuda')
        context.set_tensor_address(att_mask_name, att_mask_tensor.data_ptr())
        context.set_input_shape(att_mask_name, att_mask_shape)

        device_attention_masks[att_mask_name] = att_mask_tensor

    if len(input_info) > 2:
        token_type_name = input_info[2]["name"]
        token_type_shape = parse_shape(input_info[2]["shape"], batch_size)
        token_type_dtype = onnx_dtype_to_torch(input_info[2]["dtype"])
        token_type_tensor = torch.empty(token_type_shape, dtype=token_type_dtype, device='cuda')
        context.set_tensor_address(token_type_name, token_type_tensor.data_ptr())
        context.set_input_shape(token_type_name, token_type_shape)

        device_token_type[token_type_name] = token_type_tensor

    # Outputs vorbereiten
    for out in output_info:
        name = out["name"]
        shape = parse_shape(out["shape"], batch_size)
        dtype = onnx_dtype_to_torch(out["dtype"])  # ONNX-Datentyp in PyTorch-Datentyp umwandeln
        tensor = torch.empty(shape, dtype=dtype_out, device='cuda')
        context.set_tensor_address(name, tensor.data_ptr())
        device_outputs[name] = tensor

    device_input = next(iter(device_inputs.values()))
    device_output = next(iter(device_outputs.values()))
    # if another input: nächste attmask adresse
    if len(input_info) > 1:
        device_attention_mask = next(iter(device_attention_masks.values()))
    else:
        device_attention_mask = None

    if len(input_info) > 2:
        device_token_type = next(iter(device_token_type.values()))
    else:
        device_token_type = None

    return device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream


def build_tensorrt_engine(onnx_model_path, test_loader, batch_size, input_info=None, min_bs=1, opt_bs=8, max_bs=1024):
    """
    Erstellt und gibt die TensorRT-Engine und den Kontext zurück.
    :param onnx_model_path: Pfad zur ONNX-Modell-Datei.
    :param logger: TensorRT-Logger.
    :return: TensorRT-Engine und Execution Context.
    """
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
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 40)

    if FP16 == True:
        config.set_flag(trt.BuilderFlag.FP16)
    if INT8 == True:
        config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()

    for inp in input_info:
        name = inp["name"]
        shape = inp["shape"]
        min_shape = parse_shape(shape, min_bs)
        opt_shape = parse_shape(shape, opt_bs)
        max_shape = parse_shape(shape, max_bs)
        profile.set_shape(name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    return engine, context


def run_inference(context, test_loader, device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream, batch_size=1, input_info=None, output_info=None, accuracy_flag=False):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """


    total_predictions = 0
    correct_predictions = 0
    for i in range(2):
        for batch in test_loader: 
            # je nach Aufbau des Modells: mit Attention Mask oder ohne
            if len(batch) == 2:
                xb, yb = batch
                att_mask = None
                token_type = None
            elif len(batch) == 3:
                xb, att_mask, yb = batch
                token_type = None
            elif len(batch) == 4:
                xb, att_mask, yb, token_type = batch
            else:
                raise ValueError("Unerwartete Batch-Größe!", len(batch))
            
            dtype = onnx_dtype_to_torch(input_info[0]["dtype"])

            input_name = input_info[0]["name"]
            device_input.copy_(xb.to(dtype))
            context.set_tensor_address(input_name, device_input.data_ptr())
            context.set_input_shape(input_name, device_input.shape)

            if att_mask is not None:
                att_mask_name = input_info[1]["name"]
                device_attention_mask.copy_(att_mask.to(dtype))
                context.set_tensor_address(att_mask_name, device_attention_mask.data_ptr())
                context.set_input_shape(att_mask_name, device_attention_mask.shape)
            
            if token_type is not None:
                token_type_name = input_info[2]["name"]
                device_token_type.copy_(token_type.to(dtype))
                context.set_tensor_address(token_type_name, device_token_type.data_ptr())
                context.set_input_shape(token_type_name, device_token_type.shape)

            output_name = output_info[0]["name"]
            context.set_tensor_address(output_name, device_output.data_ptr()) 

            
            torch_stream.synchronize()

            cfx.push()
            try:
                with torch.cuda.stream(torch_stream):
                    context.execute_async_v3(stream_ptr)
            except Exception as e:
                print("TensorRT Error:", e)
            torch_stream.synchronize() 
            cfx.pop()
            end_time = time.time()

            output = device_output.cpu().numpy()


            if accuracy_flag:
                pred = output.argmax(axis=-1)  # [batch, seq_len]
                correct = (pred == yb.numpy()).sum()
                total = np.prod(yb.shape)
                correct_predictions += correct
                total_predictions += total

    accuracy = 0
    if accuracy_flag:
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return 0, 0, 0, accuracy


def start_tegrastats(logfile_path: Path):
    # tegrastats im Hintergrund starten, Ausgabe in Logdatei
    proc = subprocess.Popen(['sudo', 'tegrastats'], stdout=open(logfile_path, 'w'))
    return proc

def stop_tegrastats(proc: subprocess.Popen):
    proc.terminate()  # schickt SIGTERM an tegrastats
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

if __name__ == "__main__":
    onnx_model_path = "outputs/model_measuring.onnx"
    data_path = "data/GOLD_XYZ_OSC.0001_1024.npz"

    batch_size = 1

    if INT8:
        onnx_model_path = "outputs/model_brevitas_1.onnx"

    model = onnx.load(onnx_model_path)
    if FP16:
        model = float16.convert_float_to_float16(model)

    input_info, output_info = get_model_io_info(onnx_model_path)
    test_loader = create_test_dataloader(data_path, 1)
    engine, context = build_tensorrt_engine(onnx_model_path, test_loader, 1, input_info)
    device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream = test_data(context, 1, input_info, output_info)

    tegrastats_log = Path(__file__).resolve().parent /"energy_metrics"/ "tegrastats.log"

    # tegrastats starten
    tegra_proc = start_tegrastats(tegrastats_log)
    time.sleep(2)  # kleine Wartezeit, damit tegrastats sauber startet

    # Inferenz starten
    _, _, _, accuracy = run_inference(
        context=context,
        test_loader=test_loader,
        device_input=device_input,
        device_output=device_output,
        device_attention_mask=device_attention_mask,
        device_token_type=device_token_type,
        stream_ptr=stream_ptr,
        torch_stream=torch_stream,
        batch_size=batch_size,
        input_info=input_info,
        output_info=output_info,
        accuracy_flag=True
    )

    # tegrastats stoppen
    stop_tegrastats(tegra_proc)

    print(f"Inferenz fertig mit accuracy {accuracy:.3f}")
    print(f"tegrastats output wurde in {tegrastats_log} geschrieben.")


    del context
    del engine
    del device_input
    del device_output
    del device_attention_mask
    del device_token_type
    del torch_stream

    
    cfx.pop()  # <== explizit den Kontext entfernen!
    cfx.detach()

    parse_tegrastats_to_json.parse_tegrastats()