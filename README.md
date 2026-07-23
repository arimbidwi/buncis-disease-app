# Aplikasi Pendeteksi Penyakit Dini Daun Kacang Buncis (YOLOv11)

Web app untuk mendeteksi lokasi & jenis lesi pada daun kacang buncis
menggunakan **YOLOv11** (Ultralytics), lengkap dengan rekomendasi penanganan.

```
buncis-disease-app/
├── backend/
│   ├── main.py            # FastAPI: endpoint /predict/yolov11
│   ├── train_yolo.py       # Training YOLOv11 (Ultralytics)
│   ├── requirements.txt
│   ├── utils/disease_info.py
│   ├── models/              # Hasil model .pt disimpan di sini
│   └── dataset/             # Taruh dataset di sini (lihat langkah 2)
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

## 1. Prasyarat

- Python 3.10–3.11
- VSCode + extension Python & (opsional) Live Server
- GPU + CUDA disarankan untuk training, tapi tidak wajib

## 2. Menyiapkan dataset

Saya tidak bisa mengunduh dataset dari Kaggle/Google Drive secara langsung dari
lingkungan ini (tidak ada akses jaringan ke domain tersebut), jadi unduh manual
lalu susun sesuai struktur di bawah.

Berdasarkan folder Google Drive kamu (`KELOMPOK_BUNCIS`), datasetnya punya
**4 kategori**:

| Folder Drive | Kelas (di kode) | Kondisi |
|---|---|---|
| `EMBUN TEPUNG` | `embun_tepung` | Powdery mildew — lapisan tepung putih di daun |
| `HAWAR DAUN` | `hawar` | Leaf blight — bercak coklat kehitaman, daun mengering |
| `MOZAIK` | `mozaik` | Mosaic virus (BCMV) — pola belang hijau tua/muda |
| `SEHAT` | `sehat` | Daun normal, tanpa gejala |

YOLO butuh data beranotasi **bounding box**, bukan sekadar gambar per folder
kelas. Karena gambar di Drive kemungkinan besar masih berupa foto utuh per
kategori (format klasifikasi), perlu dianotasi dulu:

1. Unduh gambar dari tiap folder (`EMBUN TEPUNG`, `HAWAR DAUN`, `MOZAIK`, `SEHAT`) ke lokal.
2. Anotasi bounding box di sekitar area gejala (bercak/lapisan tepung/pola mosaik) memakai salah satu tool:
   - **Roboflow** (paling praktis, bisa langsung export ke format Ultralytics YOLO)
   - **LabelImg**
   - **CVAT**
3. Export dengan format **YOLO**, lalu susun ke:

```
backend/dataset/yolov11/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Isi `data.yaml`:
```yaml
path: dataset/yolov11
train: images/train
val: images/val
names:
  0: embun_tepung
  1: hawar
  2: mozaik
  3: sehat
```
(Kelas `sehat` biasanya tidak dianotasi karena tidak ada objek/lesi untuk
dideteksi — gambar sehat cukup dimasukkan tanpa file label, atau diberi 1
bounding box menutupi seluruh daun bila ingin tetap terhitung sebagai
deteksi positif "sehat".)

> **Catatan dari dataset Roboflow yang sudah kamu ekspor:** `data.yaml`
> aslinya berisi 5 kelas — ada satu kelas tambahan bernama literal `"0"` yang
> kemungkinan besar kesalahan anotasi (bukan kondisi biologis). Kode di
> `backend/utils/disease_info.py` sudah dibuat aman untuk menangani ini, tapi
> idealnya kelas tersebut dihapus/di-merge langsung di Roboflow (menu
> **Classes**) lalu dataset di-generate ulang sebelum training final.

> Alternatif cepat untuk uji coba awal: cari dataset "bean leaf disease" di
> Roboflow Universe yang sudah beranotasi bounding box, lalu export langsung
> ke format Ultralytics YOLO tanpa perlu anotasi manual dari nol.

## 3. Install dependencies backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## 4. Training model YOLOv11

```bash
cd backend
python train_yolo.py --epochs 100 --imgsz 640
# -> menghasilkan backend/models/yolov11_buncis.pt
```

Training bisa memakan waktu cukup lama di CPU; disarankan pakai GPU (Google
Colab juga bisa, lalu unduh file `best.pt` hasilnya ke `backend/models/yolov11_buncis.pt`).

## 5. Menjalankan backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Cek status model sudah siap atau belum di: `http://localhost:8000/health`

## 6. Menjalankan frontend

Buka folder `frontend/` di VSCode, lalu jalankan lewat extension **Live Server**
(klik kanan `index.html` → "Open with Live Server"), atau cara manual:

```bash
cd frontend
python -m http.server 5500
```

Lalu buka `http://localhost:5500`. Pastikan backend (`:8000`) sedang berjalan
karena frontend memanggil API tersebut (`API_BASE` di `app.js`).

## 7. Fitur antarmuka (frontend)

Frontend didesain seperti instrumen diagnostik presisi:

- **Pemindaian fleksibel** — klik untuk pilih file, seret & lepas (drag-drop), tempel gambar (Ctrl+V), atau ambil foto langsung dari kamera perangkat.
- **Overlay kotak deteksi interaktif** — kotak lesi digambar langsung di atas foto asli (bukan gambar statis dari server), warnanya mengikuti tingkat risiko kelas; arahkan kursor ke daftar deteksi untuk menyorot kotak terkait.
- **Gauge keyakinan** — indikator lingkaran menunjukkan persentase keyakinan model, warnanya berubah sesuai tingkat risiko (hijau/kuning/merah).
- **Grafik batang probabilitas** — perbandingan skor keyakinan untuk keempat kelas (Sehat, Embun Tepung, Hawar Daun, Mosaik) sekaligus, bukan cuma kelas teratas.
- **Tab hasil** — Kondisi (penyebab & gejala), Penanganan (checklist interaktif yang bisa dicentang saat langkah selesai dikerjakan), dan Deteksi (daftar objek yang ditemukan + tombol unduh gambar beranotasi).
- **Riwayat pindai** — tersimpan otomatis di `localStorage` browser (bertahan walau halaman di-refresh), tampil sebagai filmstrip thumbnail yang bisa diklik ulang untuk melihat hasil sebelumnya. Ada tombol "Hapus riwayat".
- **Notifikasi toast** — status sukses/gagal muncul sebagai notifikasi kecil, bukan `alert()` browser.

Semua fitur ini murni frontend (HTML/CSS/JS biasa, tanpa framework/build step)
sehingga bisa langsung dibuka lewat Live Server di VSCode.

## 8. Alur pemakaian aplikasi

1. Ambil/unggah/tempel foto daun kacang buncis.
2. Klik **Analisis Gambar**.
3. Hasil muncul: gauge keyakinan, label kondisi, grafik probabilitas per
   kelas, kotak deteksi di atas foto, dan tab Penanganan berisi checklist
   langkah yang bisa dicentang.

## 9. Menjadikan mobile app (opsional, lanjutan)

Frontend saat ini adalah web app responsif (bisa langsung dipakai di HP lewat
browser). Untuk membungkusnya jadi aplikasi mobile native tanpa menulis ulang:
- **Capacitor/Cordova** — bungkus `frontend/` menjadi APK/IPA langsung.
- **Flutter/React Native** — buat UI native baru yang memanggil endpoint
  FastAPI yang sama (`/predict/yolov11`).

## Catatan akurasi & etika penggunaan

Model ini adalah alat bantu skrining awal, bukan pengganti diagnosis ahli
pertanian/penyuluh lapangan. Untuk kasus serangan berat atau meragukan,
tetap konsultasikan ke penyuluh pertanian setempat.
