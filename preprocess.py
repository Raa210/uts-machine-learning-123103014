"""Fungsi pembersihan teks (dipakai saat training DAN saat prediksi di app).

Perbaikan: kata negasi (tidak, bukan, belum, ...) TIDAK dihapus sebagai stopword,
dan bentuk informal (gak, ga, gk, ...) dinormalisasi menjadi bentuk bakunya.
"""
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Kata negasi yang harus dipertahankan karena membalik makna kata sesudahnya
NEGASI = {"tidak", "tak", "bukan", "belum", "jangan", "kurang", "tanpa"}

# Bentuk informal -> bentuk baku, supaya "gak bagus" dan "tidak bagus" dianggap sama
NORMALISASI = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak",
    "enggak": "tidak", "engga": "tidak", "tdk": "tidak",
    "blm": "belum", "bkn": "bukan", "jgn": "jangan",
}

# Variasi penulisan kata "aplikasi" disatukan agar model tidak menganggapnya kata berbeda
ALIAS = {"app": "aplikasi", "apk": "aplikasi", "appnya": "aplikasi",
         "apknya": "aplikasi", "aplikasinya": "aplikasi"}

# Daftar stopword Sastrawi dikurangi kata negasi
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) - NEGASI


def bersihkan(teks: str) -> str:
    teks = str(teks).lower()                          # huruf kecil
    teks = re.sub(r"[^a-z\s]", " ", teks)             # hapus angka, tanda baca, emoji
    tokens = teks.split()                             # pecah jadi kata (sekaligus rapikan spasi)
    tokens = [NORMALISASI.get(t, t) for t in tokens]  # normalisasi gak/ga/gk -> tidak
    tokens = [ALIAS.get(t, t) for t in tokens]        # app/apk/aplikasinya -> aplikasi
    tokens = [t for t in tokens if t not in STOPWORDS]  # hapus stopword, negasi tetap ada
    return " ".join(tokens)
