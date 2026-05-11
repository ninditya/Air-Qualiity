# 🌫️ Analisis Kualitas Udara Beijing (2013–2017)

> **IDCamp 2025 — Dicoding "Belajar Analisis Data dengan Python"**
> Proyek analisis data eksploratif untuk memahami pola, tren, dan faktor pemicu polusi udara di Beijing dari 12 stasiun pemantauan selama 4 tahun, dilengkapi dashboard Streamlit interaktif.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)]()

---

## 📖 Tentang Project

Project ini merupakan analisis komprehensif terhadap **Beijing Air Quality Dataset** yang mencakup data per jam dari **12 stasiun pemantauan** selama periode **1 Maret 2013 – 28 Februari 2017** (~420.000+ baris data).

Analisis difokuskan pada **6 polutan utama** (PM2.5, PM10, SO₂, NO₂, CO, O₃) dan **6 variabel meteorologi** (temperatur, tekanan, titik embun, curah hujan, arah & kecepatan angin) untuk menjawab pertanyaan strategis terkait pengendalian polusi udara.

### 🎯 Tujuan Analisis

- 📊 Mengidentifikasi **pola temporal** polusi udara (jam kritis, hari, musim, tahun).
- 🌡️ Menganalisis **pengaruh meteorologi** terhadap konsentrasi polutan per musim.
- 🏷️ Menentukan **polutan prioritas** berdasarkan distribusi AQI overall.
- 🗺️ Memetakan **area prioritas intervensi** secara geospasial berdasarkan tingkat polusi.

---

## ❓ Pertanyaan Analisis

Empat pertanyaan utama yang dijawab dalam project ini:

| # | Fokus            | Pertanyaan                                                                                                                                                |
| - | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **Temporal**     | Bagaimana tren perbaikan kualitas udara 2013–2017 dan kapan waktu kritis (jam/hari/bulan/tahun) yang perlu diawasi?                                       |
| 2 | **Meteorologi**  | Faktor meteorologi apa yang paling berkontribusi terhadap peningkatan risiko polusi pada setiap musim?                                                    |
| 3 | **AQI Risk**     | Polutan apa yang menjadi penyebab utama penurunan kualitas udara dan bagaimana prioritas pengendaliannya? *(Clustering/Binning multi-polutan tanpa ML)*   |
| 4 | **Geospasial**   | Area mana yang memerlukan intervensi prioritas berdasarkan tingkat polusi PM2.5 dan gas polutan? *(Clustering manual tanpa ML)*                           |

---

## 📂 Struktur Direktori

```
idcamp2025-air-qualiity-beijing/
│
├── dashboard/
│   ├── dashboard.py              # Aplikasi Streamlit
│   └── main_data.csv             # Dataset hasil preprocessing untuk dashboard
│
├── data/
│   ├── air_quality_hourly_all_stations.csv   # Data per jam, 12 stasiun (gabungan)
│   └── station_coordinates.csv               # Koordinat geospasial 12 stasiun
│
├── Air-quality-dataset.zip       # Dataset mentah (raw)
├── notebook.ipynb                # Notebook analisis (versi awal)
├── notebook-revisi.ipynb         # Notebook analisis (versi final/revisi)
├── requirements.txt              # Python dependencies
├── url.txt                       # URL dashboard yang sudah di-deploy
├── .gitignore
└── README.md
```

---

## 🔄 Alur Analisis Data

Project ini mengikuti alur **Data Analysis Lifecycle** lengkap:

```
1. Gathering Data    →  Load dataset dari 12 stasiun, gabungkan ke single DataFrame
       ↓
2. Assessing Data    →  Cek missing values, duplikat, outliers, tipe data
       ↓
3. Cleaning Data     →  Handling NaN (imputation), parsing datetime, feature engineering
       ↓
4. EDA               →  Statistik deskriptif, distribusi, korelasi, pola temporal
       ↓
5. Explanatory       →  Visualisasi terarah untuk menjawab 4 pertanyaan analisis
       ↓
6. Conclusion        →  Kesimpulan + rekomendasi kebijakan
       ↓
7. Dashboard         →  Streamlit interaktif untuk eksplorasi mandiri
```

---

## ⚙️ Instalasi

### 1. Prasyarat

- Python 3.11 (disarankan)
- pip atau Anaconda
- Git

### 2. Clone Repository

```bash
git clone https://github.com/ninditya/idcamp2025-air-qualiity-beijing.git
cd idcamp2025-air-qualiity-beijing
```

### 3. Setup Virtual Environment

**Opsi A — Menggunakan Anaconda:**

```bash
conda create --name main-ds python=3.11 -y
conda activate main-ds
pip install -r requirements.txt
```

**Opsi B — Menggunakan venv (Shell/Terminal):**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

### 📓 Menjalankan Notebook Analisis

```bash
jupyter notebook notebook-revisi.ipynb
```

> 💡 Gunakan `notebook-revisi.ipynb` untuk versi final yang sudah direvisi.

### 📊 Menjalankan Dashboard Streamlit (Lokal)

```bash
streamlit run dashboard/dashboard.py
```

Dashboard akan otomatis terbuka di browser di alamat:
```
http://localhost:8501
```

---

## ☁️ Deploy ke Streamlit Community Cloud

1. Push repository ke GitHub (pastikan `dashboard/main_data.csv` ikut ter-push).
2. Buka [https://share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **New app**, isi:
   - **Repository:** repo Anda
   - **Branch:** `main`
   - **Main file path:** `dashboard/dashboard.py`
4. Klik **Deploy** dan tunggu proses build selesai.
5. Simpan URL aplikasi ke file `url.txt`.

### 🐛 Troubleshooting Deploy

| Masalah                | Solusi                                                       |
| ---------------------- | ------------------------------------------------------------ |
| Build gagal            | Cek `requirements.txt` — pastikan semua library tercantum    |
| File CSV tidak terbaca | Pastikan `dashboard/main_data.csv` ter-push ke GitHub        |
| Path entrypoint salah  | Gunakan `dashboard/dashboard.py` (bukan path lokal)          |

---

## 📊 Fitur Dashboard

Dashboard Streamlit interaktif menampilkan **metrik ringkasan** di bagian atas dan **4 tab analisis utama**:

### 🕒 Q1 — Tren & Waktu Kritis
- Historical trend polutan dari 2013–2017
- Heatmap pola jam × bulan
- Summary table per polutan
- **Insight dinamis** berbasis filter pengguna

### 🌦️ Q2 — Meteorologi & Risiko Polusi
- Rata-rata polutan per musim (seasonal averages)
- Distribusi polutan per musim (boxplot/violin)
- Korelasi musiman dengan faktor meteorologi
- Analisis pengaruh **arah angin** terhadap polusi
- Insight dinamis musim & polutan

### 🏷️ Q3 — Risiko AQI Overall
- Distribusi kategori AQI (Good/Moderate/Unhealthy/dst)
- Persentase kategori AQI per stasiun
- Ringkasan **3 stasiun terbaik & terburuk**

### 🗺️ Q4 — Geospasial
- Peta interaktif **area prioritas intervensi**
- Cluster prioritas berbasis PM2.5/AQI
- Countplot AQI per zona geografis

---

## 📈 Dataset

**Sumber:** Beijing Multi-Site Air Quality Dataset (UCI ML Repository)

| Atribut             | Detail                                                           |
| ------------------- | ---------------------------------------------------------------- |
| **Periode**         | 1 Maret 2013 — 28 Februari 2017                                  |
| **Frekuensi**       | Per jam (hourly)                                                 |
| **Jumlah Stasiun**  | 12 stasiun pemantauan di Beijing                                 |
| **Polutan**         | PM2.5, PM10, SO₂, NO₂, CO, O₃                                    |
| **Meteorologi**     | TEMP, PRES, DEWP, RAIN, WD (wind direction), WSPM (wind speed)   |
| **Total baris**     | ~420.000+ records                                                |

---

## 🛠️ Tech Stack

- **Bahasa:** Python 3.11
- **Data Analysis:** `pandas`, `numpy`
- **Visualisasi:** `matplotlib`, `seaborn`, `plotly`
- **Geospasial:** `folium`, `streamlit-folium`
- **Dashboard:** `streamlit`
- **Notebook:** `jupyter`

---

## 💡 Highlight Insight

> Beberapa temuan menarik dari analisis ini:

- 📉 **Tren membaik** — Konsentrasi PM2.5 cenderung menurun dari 2013 ke 2017, menandakan efektivitas kebijakan pengendalian.
- 🌅 **Jam kritis** — Polusi cenderung memuncak pada jam-jam tertentu (pagi & malam) terkait aktivitas transportasi & pemanasan.
- ❄️ **Musim dingin = polusi tertinggi** — Akibat pemanasan rumah berbahan bakar batu bara + inversi atmosfer.
- 🍃 **Angin = pelarut polusi** — Kecepatan angin tinggi berkorelasi negatif dengan konsentrasi PM2.5.
- 🎯 **PM2.5 = polutan prioritas** — Berkontribusi terbesar pada kategori AQI buruk.

---

## 📝 Catatan Teknis

- File `dashboard/main_data.csv` adalah hasil ekspor dari `notebook-revisi.ipynb` agar dashboard dan notebook **konsisten**.
- Koordinat stasiun di `data/station_coordinates.csv` digunakan sebagai **proxy visual geospasial** untuk perbandingan relatif antar zona.
- Analisis **tidak menggunakan Machine Learning** — clustering dilakukan secara manual via binning untuk Q3 dan Q4.

---

## 📄 Lisensi

Project ini dibuat sebagai bagian dari submission **IDCamp 2025 — Dicoding**.

---

## 👤 Author

**Ninditya Salma Nur Aini**
- GitHub: [@ninditya](https://github.com/ninditya)

---

<p align="center">
  🌍 Made with care for a cleaner future
</p>
