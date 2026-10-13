import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   
from pathlib import Path
import torch
from ultralytics import YOLO

data_dir = Path("yolo_data").resolve()
yaml_file = data_dir / "data.yaml"
text = yaml_file.read_text().splitlines()
text = [f"path: {data_dir}" if line.startswith("path:") else line for line in text]
yaml_file.write_text("\n".join(text) + "\n")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on:", device)


model = YOLO("yolo11n-seg.pt")      
model.train(
    data=str(yaml_file),
    epochs=30,          
    imgsz=640,          
    batch=8,            
    device=device,
    patience=10,        
    workers=4,
    project=str(Path("runs").resolve()),   
    name="car_damage",
    exist_ok=True,
)


best_path = model.trainer.best
best = YOLO(best_path)
metrics = best.val(data=str(yaml_file), split="test", imgsz=640, device=device)
print("Test mask mAP50:", round(metrics.seg.map50, 3))
print("Best model saved at:", best_path)