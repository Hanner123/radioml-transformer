import json
from pathlib import Path

# Pfade zu den einzelnen Accuracy-Dateien
base_dir = Path(__file__).resolve().parent / "eval_results"
files = [
    base_dir / "accuracy_FP16.json",
    base_dir / "accuracy_FP32.json",
    base_dir / "accuracy_INT8.json",
    base_dir / "accuracy_INT8_tensorrt.json",
]

# Einträge laden
results = []
for file in files:
    if file.exists():
        with open(file, "r") as f:
            data = json.load(f)
            # Falls die Datei eine Liste enthält, erweitern, sonst anhängen
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)

# Zusammengefasste Datei speichern
output_path = base_dir / "accuracy.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Zusammengefasste Datei gespeichert unter: {output_path}")