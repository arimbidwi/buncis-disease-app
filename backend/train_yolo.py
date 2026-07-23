"""
Skenario 2: Melatih model deteksi objek YOLOv11 (ultralytics) untuk
melokalisasi dan mengklasifikasi lesi/gejala penyakit pada daun kacang buncis.

YOLO butuh anotasi bounding box (format YOLO .txt), BUKAN dataset klasifikasi
folder-per-kelas. Jika dataset sumber (Kaggle/Drive) hanya berupa gambar utuh
per kelas, gunakan tool anotasi seperti Roboflow, LabelImg, atau CVAT untuk
membuat bounding box di sekitar area bercak/lesi sebelum training di sini.

Struktur dataset yang diharapkan (format Ultralytics YOLO), letakkan di:
backend/dataset/yolov11/

dataset/yolov11/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

Contoh isi data.yaml (4 kategori sesuai folder dataset: EMBUN TEPUNG,
HAWAR DAUN, MOZAIK, SEHAT):
    path: dataset/yolov11
    train: images/train
    val: images/val
    names:
      0: embun_tepung
      1: hawar
      2: mozaik
      3: sehat

(Catatan: kelas 'sehat' biasanya tidak dianotasi karena tidak ada objek/lesi
untuk dideteksi — gambar sehat cukup dimasukkan tanpa file label, atau
dianotasi dengan 1 bounding box menutupi seluruh daun bila ingin tetap
dihitung sebagai deteksi positif "sehat".)

Jalankan:
    python train_yolo.py --epochs 100 --imgsz 640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATASET_YAML = BASE_DIR / "dataset" / "yolov11" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                         help="Bobot awal pretrained YOLOv11 (n/s/m/l/x)")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        raise SystemExit(
            f"File konfigurasi dataset tidak ditemukan di {DATASET_YAML}.\n"
            "Siapkan dataset beranotasi bounding box (format YOLO) dan buat "
            "data.yaml sesuai contoh pada docstring skrip ini."
        )

    model = YOLO(args.weights)  # otomatis mengunduh bobot pretrained bila belum ada
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(MODELS_DIR / "runs"),
        name="yolov11_buncis",
    )

    metrics = model.val()
    print("Hasil validasi:", metrics.results_dict)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_weights = MODELS_DIR / "runs" / "yolov11_buncis" / "weights" / "best.pt"
    target_path = MODELS_DIR / "yolov11_buncis.pt"
    if best_weights.exists():
        target_path.write_bytes(best_weights.read_bytes())
        print(f"Model terbaik disalin ke: {target_path}")
    else:
        print(f"Peringatan: {best_weights} tidak ditemukan, cek folder runs/ secara manual.")


if __name__ == "__main__":
    main()
