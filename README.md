# Car Damage Detection

Find and label damage on a car from a single photo.

This project trains a **YOLO11 instance segmentation** model on the **CarDD** dataset. A **Streamlit** web app then lets anyone upload a car photo and see each damaged area outlined and named.

---

## What it detects

The model finds 6 types of damage:

| Class (dataset label) | Shown in the app as |
| --- | --- |
| `dent` | Dent (panel damage) |
| `scratch` | Scratch |
| `crack` | Crack |
| `glass shatter` | Broken glass |
| `lamp broken` | Broken lamp |
| `tire flat` | Flat tyre |

For each photo, the app shows:

- the image with each damaged area outlined (segmentation mask + box)
- a count of each damage type found
- a confidence score for every detection

---

## How it works

```
CarDD (COCO format)  →  convert_to_yolo.py  →  yolo_data/  →  train.py  →  best.pt  →  app.py
   polygons + JSON         YOLO seg labels       images +       YOLO11n-seg    trained    Streamlit
                                                 data.yaml      fine-tuning    weights    web app
```

1. **Data conversion** (`convert_to_yolo.py`)
   Reads the CarDD COCO annotation files for the `train`, `val` and `test` splits. Each damage polygon is scaled to 0–1 and written as a YOLO segmentation label. Images are linked (not copied) into `yolo_data/images/`, which saves disk space. It also writes `yolo_data/data.yaml` with the class names.

2. **Training** (`train.py`)
   Fine-tunes the pretrained `yolo11n-seg.pt` model. It uses the Apple Silicon GPU (`mps`) when available, and falls back to CPU otherwise. After training, it evaluates the best model on the **test** split and prints the mask mAP50.

3. **App** (`app.py`)
   Loads the trained weights once (cached), runs prediction on the uploaded photo, and shows the results.

---

## Tech stack

- Python
- [Ultralytics YOLO11](https://docs.ultralytics.com/) (instance segmentation)
- PyTorch (MPS / CPU)
- Streamlit
- Pillow

---

## Project structure

```
Car-Damage-Detection/
├── convert_to_yolo.py      # CarDD COCO → YOLO segmentation format
├── train.py                # Train + test the YOLO11n-seg model
├── app.py                  # Streamlit web app
├── dataset/
│   └── extracted/CarDD_release/CarDD_COCO/   # CarDD dataset (not included)
├── yolo_data/              # Created by convert_to_yolo.py
└── runs/car_damage/weights/best.pt           # Created by train.py
```

---

## Getting started

### 1. Install

```bash
git clone https://github.com/Rishi-Kora/Car-Damage-Detection.git
cd Car-Damage-Detection
pip install ultralytics streamlit pillow torch
```

### 2. Get the dataset

This project uses **CarDD: A New Dataset for Vision-based Car Damage Detection** ([paper](https://arxiv.org/abs/2211.00945)). The dataset is not included in this repo. Request it from the [official CarDD page](https://cardd-ustc.github.io) (non-commercial research and educational use only) and extract it so this folder exists:

```
dataset/extracted/CarDD_release/CarDD_COCO/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
├── train2017/
├── val2017/
└── test2017/
```

### 3. Convert the data

```bash
python convert_to_yolo.py
```

### 4. Train the model

```bash
python train.py
```

The best model is saved to `runs/car_damage/weights/best.pt`.

### 5. Run the app

```bash
streamlit run app.py
```

Open the link shown in the terminal, upload a `.jpg`, `.jpeg` or `.png` photo, and see the results.

---

## Training settings

| Setting | Value |
| --- | --- |
| Base model | `yolo11n-seg.pt` (pretrained) |
| Epochs | 30 |
| Early stopping patience | 10 epochs |
| Image size | 640 |
| Batch size | 8 |
| Device | `mps` (Apple Silicon) or `cpu` |
| App confidence threshold | 0.25 |

---

## Results

| Metric (test split) | Value |
| --- | --- |
| Mask mAP50 | _add the value printed at the end of `train.py`_ |

---

## Acknowledgements

- **CarDD dataset:** Wang, X., Li, W., & Wu, Z. — *CarDD: A New Dataset for Vision-based Car Damage Detection* ([arXiv:2211.00945](https://arxiv.org/abs/2211.00945)). Please follow the dataset's own licence and terms of use.
- **Ultralytics** for the YOLO11 models and training tools.

---

## Author

**Rishi Kora** — [GitHub](https://github.com/Rishi-Kora)
