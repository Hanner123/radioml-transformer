import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import torch
import onnx
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pycuda.driver as cuda
import pycuda.autoinit
import os
import yaml
from onnxconverter_common import float16
# tensorrt, datasets(hugging face), pycuda
FP16 = os.environ.get("FP16", "0") == "1"
if FP16:
    dtype = torch.float16
    print("FP16 enabled")
else:
    dtype = torch.float32
    print("FP32")

def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
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
    

# accuracy vom LLM berechnen

def accuracy(labels, outputs):
    # funktioniert nicht mit größerer batch size
    correct_predictions = 0
    total_predictions = 0
    i = 0
    for label in labels: 
        predicted = np.argmax(outputs, axis=1)
        total_predictions = total_predictions + 1
        if predicted == label:
            correct_predictions = correct_predictions + 1
        i = i+1
    return correct_predictions, total_predictions

def save_json(log, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)




def measure_latency(context, test_loader, device_input, device_output, stream_ptr, torch_stream, batch_size=1):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """
    total_time = 0
    total_time_synchronize = 0
    total_time_datatransfer = 0  # Gesamte Laufzeit aller gemessenen Batches
    iterations = 0  # Anzahl gemessener Batches
    # wie kann ich die input-sätze von dem Dataloader in den device_input buffer laden?
    for xb, yb in test_loader:  
        start_time_datatransfer = time.time()  # Startzeit messen
        # print("xb:", xb.shape, xb.dtype)
        # print("device_input:", device_input.shape, device_input.dtype)
        # Buffer-Addresses und Shape JEDES MAL neu setzen!


        device_input.copy_(xb.to(dtype))

        context.set_tensor_address("input", device_input.data_ptr()) #important for fp16 version... i dont know why
        context.set_tensor_address("output", device_output.data_ptr())
        context.set_input_shape("input", device_input.shape)

        start_time_synchronize = time.time()  # Startzeit messen
        torch_stream.synchronize()  

        start_time_inteference = time.time()  # Startzeit messen
        try:
            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
        except Exception as e:
            print("TensorRT Error:", e)
        torch_stream.synchronize()  # GPU-Synchronisation nach Inferenz
        end_time = time.time()

        output = device_output.cpu().numpy()
        end_time_datatransfer = time.time() 

        latency = end_time - start_time_inteference  # Latenz für diesen Batch
        latency_synchronize = end_time - start_time_synchronize  # Latenz für diesen Batch
        latency_datatransfer = end_time_datatransfer - start_time_datatransfer  # Latenz für diesen Batch

        total_time += latency
        total_time_synchronize += latency_synchronize
        total_time_datatransfer += latency_datatransfer
        iterations += 1
        
        # labels auswerten - zeit messen, bar plots
    average_latency = (total_time / iterations) * 1000  # In Millisekunden
    average_latency_synchronize = (total_time_synchronize / iterations) * 1000  # In Millisekunden
    average_latency_datatransfer = (total_time_datatransfer / iterations) * 1000  # In Millisekunden


    return average_latency, average_latency_synchronize, average_latency_datatransfer

def print_latency(latency_ms, latency_synchronize, latency_datatransfer, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size):
    print("For Batch Size: ", batch_size)
    print(f"Gemessene durchschnittliche Latenz für Inteferenz : {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation : {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer : {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time-start_time:.4f} s")
    print("num_batches", num_batches)
    print(f"Throughput: {throughput_batches:.4f} Batches/Sekunde")
    print(f"Throughput: {throughput_images:.4f} Bilder/Sekunde")





def create_test_dataloader(data_path, batch_size, seq_len=32, emb_dim=64):
    import h5py
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    with h5py.File(data_path, "r") as f:
        X = np.array(f["X"][:10000])  # Nur die ersten 1000 Datensätze
        Y = np.array(f["Y"][:10000])

    # Reshape wie im Training!
    # Beispiel: von [samples, 1024, 2] zu [samples, 32, 64]
    # Dazu erst auf [samples, 1024*2], dann auf [-1, 32, 64]
    X = X.reshape(X.shape[0], -1)           # [samples, 2048]
    X = X.reshape(-1, seq_len, emb_dim)     # [samples', 32, 64]

    # Labels ggf. anpassen (z.B. argmax, expand, ... wie im Training)
    if Y.ndim == 2 and Y.shape[1] > 1:
        Y = np.argmax(Y, axis=1)
    Y = np.tile(Y[:, None], (1, seq_len))   # [samples', seq_len]

    X = torch.tensor(X, dtype=dtype) #dtype
    Y = torch.tensor(Y, dtype=torch.long)
    test_dataset = TensorDataset(X, Y)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    return test_loader

def calculate_latency_and_throughput(context, batch_sizes):
    """
    Berechnet die durchschnittliche Latenz und den Durchsatz (Bilder und Batches pro Sekunde) für verschiedene Batchgrößen.
    :param context: TensorRT-Execution-Context.
    :param test_loader: DataLoader mit Testdaten.
    :param device_input: Eingabebuffer auf der GPU.
    :param device_output: Ausgabebuffer auf der GPU.
    :param stream_ptr: CUDA-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param batch_sizes: Liste der Batchgrößen.
    :return: (Throughput-Log, Latenz-Log).
    """
    

    throughput_log = []
    latency_log = []
    latency_log_batch = []

    for batch_size in batch_sizes:
        test_loader = create_test_dataloader(data_path, batch_size) 
        engine_name = f"radioml_int8_{batch_size}.engine"
        engine_path = Path(__file__).resolve().parent / "outputs" / "engines" / engine_name
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        device_input, device_output, stream_ptr, torch_stream = test_data(context, batch_size)

        
        # Schleife für durchschnitt
        latency_ms_sum = 0
        latency_synchronize_sum = 0
        lantency_datatransfer_sum = 0
        total_time_sum = 0
        num_executions = 10.0
        for i in range(int(num_executions)):
            start_time = time.time()
            latency_ms, latency_synchronize, latency_datatransfer = measure_latency(
                context=context,
                test_loader=test_loader,
                device_input=device_input,
                device_output=device_output,
                stream_ptr=stream_ptr,
                torch_stream=torch_stream,
                batch_size=batch_size
            )
            latency_ms_sum = latency_ms_sum + latency_ms
            latency_synchronize_sum = latency_synchronize_sum + (latency_synchronize-latency_ms)
            lantency_datatransfer_sum = lantency_datatransfer_sum + (latency_datatransfer-latency_synchronize)

            end_time = time.time()
            total_time_sum = total_time_sum + (end_time-start_time)


        latency_avg = float(latency_ms_sum/num_executions)
        latency_synchronize_avg = float(latency_synchronize_sum/num_executions)
        latency_datatransfer_avg = float(lantency_datatransfer_sum/num_executions)
        total_time_avg = float(total_time_sum/num_executions)

        num_batches = int(7600/batch_size) 
        throughput_batches = num_batches/(total_time_avg) 
        throughput_images = (num_batches*batch_size)/(total_time_avg)


        log_latency_inteference = {"batch_size": batch_size, "type":"inteference", "value": latency_avg/batch_size} # pro datensatz?
        log_latency_synchronize = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize_avg/batch_size)} # pro datensatz?
        log_latency_datatransfer = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer_avg/batch_size)} # pro datensatz?
        log_latency_inteference_batch = {"batch_size": batch_size, "type":"inteference", "value": latency_avg} #pro batch
        log_latency_synchronize_batch = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize_avg)} #pro batch
        log_latency_datatransfer_batch = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer_avg)} #pro batch 
        throughput = {"batch_size": batch_size, "throughput_images_per_s": throughput_images, "throughput_batches_per_s": throughput_batches}


        throughput_log.append(throughput)
        latency_log.extend([log_latency_inteference, log_latency_synchronize, log_latency_datatransfer])
        latency_log_batch.extend([log_latency_inteference_batch, log_latency_synchronize_batch, log_latency_datatransfer_batch])
        print_latency(latency_avg, latency_synchronize_avg+latency_avg, latency_datatransfer_avg+latency_synchronize_avg+latency_avg, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size)

    return throughput_log, latency_log, latency_log_batch

def test_data(context, batch_size):
    input_name = "input"    # Name wie im ONNX-Modell
    output_name = "output"  # Name wie im ONNX-Modell
    input_shape = (batch_size, 32, 64)
    output_shape = (batch_size, 32, 24)
    device_input = torch.empty(input_shape, dtype=dtype, device='cuda')
    device_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream
    context.set_tensor_address(input_name, device_input.data_ptr())
    context.set_tensor_address(output_name, device_output.data_ptr())
    context.set_input_shape(input_name, input_shape)  # für dynamische batch size
    return device_input, device_output, stream_ptr, torch_stream


def append_json(new_entry, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Bestehende Daten laden, falls vorhanden
    if filepath.exists():
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except Exception:
                data = []
    else:
        data = []
    data.append(new_entry)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def run_inference(batch_size=1):
    """pynvml-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param max_iterations: Maximalanzahl der Iterationen.
    :return: (Anzahl der korrekten Vorhersagen, Gesamtanzahl der Vorhersagen).
    """
    engine_name = f"radioml_int8_32.engine"
    engine_path = Path(__file__).resolve().parent / "outputs" / "engines" / engine_name
    test_loader = create_test_dataloader(data_path, batch_size)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    device_input, device_output, stream_ptr, torch_stream = test_data(context, batch_size) # anpassen!!
    print("device_input:", device_input.shape, device_input.dtype)
    print("device_output:", device_output.shape, device_output.dtype)

    total_predictions = 0
    correct_predictions = 0


    for xb, yb in test_loader:

        device_input.copy_(xb.to(dtype))

        context.set_tensor_address("input", device_input.data_ptr())
        context.set_tensor_address("output", device_output.data_ptr())
        context.set_input_shape("input", device_input.shape)
        torch_stream.synchronize()
        
        try:
            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
        except Exception as e:
            print("TensorRT Error:", e)
        torch_stream.synchronize()
        torch.cuda.synchronize()  # Warten auf Abschluss der Inferenz
        output = device_output.cpu().numpy()

        pred = output.argmax(axis=-1)  # [batch, seq_len]
        correct = (pred == yb.numpy()).sum()
        total = np.prod(yb.shape)
        correct_predictions += correct
        total_predictions += total
    # del device_input, device_output, stream_ptr, torch_stream, engine, context
    return correct_predictions, total_predictions

# Montag:
# richtiges modell verwenden/an passender Stelle exportieren - funkioniert nicht, schon in model.py sind nodes die tensorrt nicht versteht
# erstmal eigenes, ähnliches modell verwenden. die outputs und inputs sind aber gleich
# in pipeline einbauen - gemacht
# grafiken erstellen - gemacht


# Dienstag:
# FP 16 testen - gemacht, aber kein unterschied zu FP 32 (etwas schlechter), [06/17/2025-12:29:42] [TRT] [E] IExecutionContext::enqueueV3: Error Code 1: Cuda Runtime (invalid resource handle)
# quantisiertes modell messen, testen, ähnlicheres modell verwenden


if __name__ == "__main__":

    data_path = "data/GOLD_XYZ_OSC.0001_1024.hdf5"  # Pfad zu den Testdaten

    batch_sizes = [1,2,4,8, 16, 32, 64, 128, 256, 512, 1024]  # Liste der Batchgrößen


    context=0
    correct_predictions, total_predictions = run_inference(batch_size=1)  # Teste Inferenz mit Batch Size 1
    print(f"Accuracy : {correct_predictions / total_predictions:.2%}")

    accuracy_result = {
        "quantisation_type": "INT8 TensorRT",
        "value": correct_predictions / total_predictions
    }
    accuracy_path = Path(__file__).resolve().parent / "eval_results" /"accuracy_INT8_tensorrt.json"
    save_json(accuracy_result, accuracy_path)
    


    throughput_log, latency_log, latency_log_batch = calculate_latency_and_throughput(context, batch_sizes)
    throughput_results = Path(__file__).resolve().parent / "throughput" / "INT8_tensorrt" /"throughput_results.json"
    throughput_results2 = Path(__file__).resolve().parent / "throughput"/ "INT8_tensorrt" / "throughput_results_2.json"
    latency_results = Path(__file__).resolve().parent / "throughput" / "INT8_tensorrt"/ "latency_results.json"
    latency_results_batch = Path(__file__).resolve().parent / "throughput" / "INT8_tensorrt"/ "latency_results_batch.json"
    save_json(throughput_log, throughput_results)
    save_json(throughput_log, throughput_results2)
    save_json(latency_log, latency_results)
    save_json(latency_log_batch, latency_results_batch)

