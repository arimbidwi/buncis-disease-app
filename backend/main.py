"""
Backend API - Aplikasi Pendeteksi Penyakit Dini Daun Kacang Buncis
Menggunakan YOLOv11 (ultralytics) untuk mendeteksi lokasi & jenis lesi pada daun.

Jalankan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.disease_info import CLASS_NAMES, get_info

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov11_buncis.pt"

app = FastAPI(title="Deteksi Penyakit Daun Kacang Buncis API (YOLOv11)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan origin frontend spesifik saat produksi
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model dimuat sekali (lazy) saat endpoint pertama kali dipanggil, supaya
# server tetap bisa start walau model belum dilatih/diletakkan di folder models/.
# ---------------------------------------------------------------------------
_yolo_model = None


def _load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model YOLOv11 belum ditemukan di {YOLO_MODEL_PATH}. "
                    "Jalankan backend/train_yolo.py terlebih dahulu untuk melatih "
                    "dan menyimpan model (best.pt)."
                ),
            )
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar yang valid.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolov11_model_ready": YOLO_MODEL_PATH.exists(),
    }


@app.post("/predict/yolov11")
async def predict_yolov11(file: UploadFile = File(...)):
    """Deteksi lokasi & jenis lesi daun memakai YOLOv11."""
    model = _load_yolo_model()
    img = _read_image(await file.read())
    img_w, img_h = img.size

    results = model.predict(source=np.array(img), verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    # Gambar hasil anotasi (kotak deteksi) dikirim balik sebagai base64,
    # dipakai untuk tombol "unduh hasil" di frontend.
    annotated = result.plot()  # numpy array BGR
    annotated_rgb = annotated[:, :, ::-1]
    annotated_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG", quality=90)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Ringkasan keyakinan per kelas (nilai tertinggi antar deteksi sekelas),
    # dipakai frontend untuk grafik batang perbandingan.
    class_confidences = {c: 0.0 for c in CLASS_NAMES}
    for d in detections:
        key = d["label"].lower().strip().replace(" ", "_")
        if key in class_confidences:
            class_confidences[key] = max(class_confidences[key], d["confidence"])

    if detections:
        top = max(detections, key=lambda d: d["confidence"])
        label_utama = top["label"]
        confidence = top["confidence"]
        info = get_info(label_utama)
    else:
        label_utama = "sehat"
        confidence = 1.0
        class_confidences["sehat"] = 1.0
        info = get_info("sehat")

    return JSONResponse({
        "deteksi": detections,
        "label_utama": label_utama,
        "confidence": confidence,
        "probabilitas_kelas": class_confidences,
        "gambar_anotasi_base64": annotated_b64,
        "gambar_ukuran": {"width": img_w, "height": img_h},
        "info": info,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
