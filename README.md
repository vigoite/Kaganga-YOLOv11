# Deteksi Aksara Ulu Rejang Menggunakan YOLO11n

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem deteksi objek untuk mengenali karakter **Aksara Ulu Rejang (Kaganga)** menggunakan model **YOLO11n** dari framework Ultralytics. Dataset diperoleh dari Roboflow dan digunakan untuk melatih model agar mampu mendeteksi berbagai karakter aksara pada gambar secara otomatis.

Model dilatih menggunakan pendekatan object detection sehingga setiap karakter dapat dikenali sekaligus ditentukan posisi (bounding box)-nya pada gambar.

---

## Tujuan Proyek

- Membangun model deteksi karakter Aksara Ulu Rejang berbasis Deep Learning.
- Mengimplementasikan YOLO11n untuk pengenalan aksara tradisional.
- Mengevaluasi performa model menggunakan metrik object detection.
- Menghasilkan model yang dapat digunakan untuk identifikasi aksara secara otomatis.

---

## Fitur dan Konfigurasi Training

### Framework

- Python 3.0
- Ultralytics YOLO11
- Roboflow
- Google Colab

### Model

- YOLO11n (Nano)
- Pre-trained COCO
- Fine-tuning pada dataset Aksara Ulu Rejang

### Parameter Training

| Parameter | Nilai |
|------------|--------|
| Epoch | 200 |
| Image Size | 640x640 |
| Batch Size | 16 |
| Patience | 20 |
| Task | Object Detection |
| Dataset Format | YOLOv8 |

```python
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    patience=20,
    name="kaganga_yolo11n"
)
```

---

## Persiapan dan Instalasi

### Clone Repository

```bash
git clone https://github.com/username/kaganga-yolo11.git
cd kaganga-yolo11
```

### Install Dependensi

```bash
pip install ultralytics roboflow
```

### Import Library

```python
from ultralytics import YOLO
from roboflow import Roboflow
```

### Konfigurasi API Roboflow

```python
api_key = "YOUR_ROBOFLOW_API_KEY"
```

### Download Dataset

```python
rf = Roboflow(api_key=api_key)

project = rf.workspace(
    "novalrizkiansyah-ymail-com"
).project(
    "aksara-ulu-rejang"
)

dataset = project.version(4).download("yolov8")
```

---

## Struktur Dataset

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
├── data.yaml
└── README.dataset.txt
```

---

## Cara Menjalankan Training

### Muat Model

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
```

### Jalankan Training

```python
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    patience=20,
    name="kaganga_yolo11n"
)
```

---

## Hasil Training

### Evaluasi Model

```python
metrics = model.val(
    data=f"{dataset.location}/data.yaml",
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    plots=True
)
```

### Output yang Dihasilkan

- Precision
- Recall
- mAP50
- mAP50-95
- F1-Score
- Confusion Matrix
- Precision-Recall Curve

### Model Terbaik

```text
runs/detect/kaganga_yolo11n/weights/best.pt
```

---

## Prediksi

```python
model.predict(
    source="gambar_uji.jpg",
    conf=0.25,
    save=True
)
```

Hasil prediksi akan disimpan pada:

```text
runs/detect/predict/
```

---

## Anggota Kelompok
1. Fanni Ghina Athiyyah (G1A022087)
2. Vigo Ite Anugrahesa  (G1A022089)

