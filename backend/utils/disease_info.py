"""
Informasi kelas kondisi daun kacang buncis beserta rekomendasi penanganan.
Kelas mengikuti persis nama pada dataset Roboflow (lihat backend/dataset/yolov11/data.yaml):
  - embun tepung  (Powdery Mildew)
  - hawar         (Leaf Blight / Hawar Daun)
  - mozaik        (Mosaic Virus)
  - sehat         (Healthy)

Catatan: dataset sumber sempat memuat kelas tambahan bernama literal "0" di
posisi index pertama (nc: 5). Ini kemungkinan besar kesalahan saat anotasi di
Roboflow (bukan kelas biologis). Sebaiknya dihapus/di-merge langsung di
Roboflow lalu dataset di-generate ulang. Selama belum diperbaiki di sumber,
kode di sini tetap menanganinya dengan aman (lihat get_info di bawah).
"""

CLASS_NAMES = ["embun_tepung", "hawar", "mozaik", "sehat"]

DISEASE_INFO = {
    "embun_tepung": {
        "nama": "Embun Tepung (Powdery Mildew)",
        "penyebab": "Jamur Erysiphe polygoni / Podosphaera sp.",
        "kondisi": (
            "Muncul lapisan tepung berwarna putih keabu-abuan di permukaan "
            "daun (atas maupun bawah), batang muda, dan kadang polong. "
            "Daun yang terserang berat akan menguning, mengeriting, lalu "
            "gugur lebih awal. Berkembang pesat pada kondisi lembap dengan "
            "sirkulasi udara buruk, terutama saat siang panas-malam sejuk."
        ),
        "penanganan": [
            "Pangkas dan musnahkan bagian tanaman yang terserang berat agar spora tidak menyebar lewat angin.",
            "Semprotkan fungisida berbahan aktif sulfur, belerang, atau triazol sesuai dosis anjuran, ulangi setiap 7-10 hari.",
            "Perbaiki jarak tanam dan lakukan pemangkasan daun rimbun untuk meningkatkan sirkulasi udara.",
            "Sirami tanaman di pagi hari agar daun cepat kering, hindari penyiraman di sore/malam hari.",
            "Rotasi tanaman dan hindari penanaman berulang di lahan yang sama pada musim berikutnya.",
        ],
        "tingkat_risiko": "sedang",
    },
    "hawar": {
        "nama": "Hawar Daun (Leaf Blight)",
        "penyebab": "Bakteri Xanthomonas axonopodis pv. phaseoli / jamur patogen tular tanah",
        "kondisi": (
            "Bercak coklat kehitaman dengan tepi kekuningan (halo) muncul "
            "di tepi atau tengah daun, meluas dan menyatu hingga daun "
            "mengering dan hangus seperti terbakar (blight). Menyebar cepat "
            "lewat percikan air hujan/irigasi dan kelembapan tinggi."
        ),
        "penanganan": [
            "Cabut dan musnahkan (bakar/kubur) tanaman yang terserang berat agar tidak jadi sumber infeksi baru.",
            "Semprotkan bakterisida/fungisida berbahan aktif tembaga (copper-based) sesuai dosis anjuran.",
            "Hindari penyiraman dengan cara memercik ke daun; gunakan irigasi tetes atau siram di pangkal batang.",
            "Sanitasi alat pertanian dan tangan setelah kontak dengan tanaman sakit untuk mencegah penularan.",
            "Gunakan benih sehat/bersertifikat dan lakukan rotasi tanaman non-legum minimal 1 musim.",
        ],
        "tingkat_risiko": "tinggi",
    },
    "mozaik": {
        "nama": "Mosaik (Bean Common Mosaic Virus)",
        "penyebab": "Virus BCMV (Bean Common Mosaic Virus), ditularkan oleh kutu daun (aphid)",
        "kondisi": (
            "Daun menunjukkan pola belang hijau tua-hijau muda/kuning "
            "(mosaik), permukaan daun bergelombang atau mengeriting, dan "
            "pertumbuhan tanaman menjadi kerdil dengan hasil polong menurun. "
            "Tidak dapat disembuhkan setelah terinfeksi karena bersifat virus."
        ),
        "penanganan": [
            "Cabut dan musnahkan tanaman yang terinfeksi sesegera mungkin untuk memutus sumber penularan virus.",
            "Kendalikan populasi kutu daun (vektor) dengan insektisida nabati atau kimia sesuai anjuran.",
            "Gunakan mulsa plastik perak (silver) untuk mengusir kutu daun secara mekanis.",
            "Gunakan benih tahan virus (varietas resisten BCMV) pada musim tanam berikutnya.",
            "Hindari menanam berdekatan dengan tanaman legum lain yang sudah terinfeksi.",
        ],
        "tingkat_risiko": "tinggi",
    },
    "sehat": {
        "nama": "Sehat",
        "penyebab": "-",
        "kondisi": (
            "Daun berwarna hijau merata, tanpa bercak, lapisan tepung, "
            "maupun pola belang. Pertumbuhan tanaman normal dan tidak ada "
            "gejala kelainan."
        ),
        "penanganan": [
            "Lanjutkan perawatan rutin: penyiraman teratur, pemupukan berimbang, dan sanitasi kebun.",
            "Lakukan pemantauan berkala (idealnya mingguan) untuk deteksi dini apabila muncul gejala baru.",
            "Jaga jarak tanam dan drainase lahan agar kondisi tetap optimal dan kelembapan tidak berlebih.",
        ],
        "tingkat_risiko": "rendah",
    },
    # Kelas anomali dari dataset sumber (lihat catatan di atas file ini).
    "0": {
        "nama": "Kelas Tidak Valid",
        "penyebab": "-",
        "kondisi": (
            "Label ini bukan kondisi daun yang sebenarnya — kemungkinan besar "
            "kesalahan anotasi pada dataset sumber (Roboflow). Disarankan "
            "menghapus/menggabungkan kelas ini di Roboflow lalu men-generate "
            "ulang dataset sebelum training berikutnya."
        ),
        "penanganan": ["Periksa dan bersihkan anotasi kelas ini di Roboflow, lalu latih ulang model."],
        "tingkat_risiko": "-",
    },
}


def get_info(label: str) -> dict:
    key = label.lower().strip().replace(" ", "_")
    return DISEASE_INFO.get(key, {
        "nama": label,
        "penyebab": "Tidak diketahui",
        "kondisi": "Tidak ada data tambahan untuk kelas ini.",
        "penanganan": ["Konsultasikan dengan penyuluh pertanian setempat."],
        "tingkat_risiko": "-",
    })
