# Proyek Analisis Data: Air Quality Beijing (2013-2017)

Submission untuk kelas Dicoding **Belajar Analisis Data dengan Python** menggunakan **Air Quality Dataset** (12 stasiun, data per jam, 1 Maret 2013 - 28 Februari 2017).

## Cakupan Proyek

Proyek ini memenuhi alur analisis lengkap:
- Gathering data
- Assessing data
- Cleaning data
- Exploratory Data Analysis (EDA)
- Explanatory analysis
- Conclusion + rekomendasi
- Dashboard Streamlit interaktif

Empat pertanyaan analisis yang dijawab:
1. Q1 (Temporal): Pada periode 2013-2017, bagaimana tren perbaikan kualitas udara dan kapan waktu kritis (jam, hari, bulan, tahun) yang perlu menjadi fokus pengawasan polutan?
2. Q2 (Meteorologi): Pada setiap musim dalam periode 2013-2017, faktor meteorologi apa yang paling berkontribusi terhadap peningkatan risiko polusi?
3. Q3 (Clustering/Binning AQI Multi-Polutan tanpa ML): Berdasarkan data periode 2013-2017, polutan apa yang menjadi penyebab utama penurunan kualitas udara dan bagaimana prioritas pengendaliannya berdasarkan tingkat risiko?
4. Q4 (Geospasial + Clustering Manual tanpa ML): Pada periode 2013-2017, area mana yang memerlukan intervensi prioritas berdasarkan tingkat polusi PM2.5 dan gas polutan?

## Struktur Direktori

```text
submission-final/
├── dashboard/
│   ├── dashboard.py
│   └── main_data.csv
├── data/
│   ├── air_quality_hourly_all_stations.csv
│   └── station_coordinates.csv
├── notebook-revisi.ipynb
├── requirements.txt
├── README.md
└── url.txt
```

## Menjalankan Proyek

### Menyiapkan Lingkungan (Anaconda)
```bash
conda create --name main-ds python=3.11 -y
conda activate main-ds
pip install -r requirements.txt
```

### Menyiapkan Lingkungan (Shell/Terminal)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Jalankan Notebook
```bash
jupyter notebook notebook-revisi.ipynb
```

### Jalankan Dashboard Streamlit
```bash
streamlit run dashboard/dashboard.py
```

### Deploy ke Streamlit Community Cloud
1. Buat repository GitHub, lalu upload isi folder proyek ini (pastikan `dashboard/main_data.csv` ikut ter-push).
2. Buka [https://share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **New app** lalu isi:
   - **Repository**: repo proyek Anda
   - **Branch**: `main` (atau branch yang dipakai)
   - **Main file path**: `dashboard/dashboard.py`
4. Klik **Deploy** dan tunggu proses build selesai.
5. Jika sukses, salin URL app dan isi ke file `url.txt`.

Contoh isi `url.txt`:
```text
https://nama-app-anda.streamlit.app
```

Jika build gagal:
- Pastikan nama file entrypoint tepat: `dashboard/dashboard.py`
- Pastikan dependensi ada di `requirements.txt`
- Pastikan data ada di `dashboard/main_data.csv` (bukan path lokal komputer)

## Navigasi Dashboard

Dashboard menampilkan ringkasan metrik di bagian atas dan mencakup 4 fokus analisis utama:
1. **Q1 - Tren dan Waktu Kritis**: history trend, heatmap jam-bulan, summary table, dan insight dinamis berbasis filter.
2. **Q2 - Meteorologi dan Risiko Polusi**: season averages, seasonal distribution, korelasi musiman dengan faktor meteorologi, arah angin, dan insight dinamis.
3. **Q3 - Risiko AQI Overall**: distribusi kategori AQI overall, persentase kategori per stasiun, serta ringkasan 3 stasiun terbaik/terburuk.
4. **Q4 - Geospasial**: peta area prioritas berbasis PM2.5/AQI dengan cluster prioritas, dilengkapi countplot AQI per zona.

## Catatan

- File `dashboard/main_data.csv` diekspor dari notebook agar analisis notebook dan dashboard konsisten.
- Koordinat stasiun dipakai sebagai proxy visual geospasial untuk perbandingan relatif antarzona.
- Jika dashboard sudah dideploy ke Streamlit Community Cloud, isi `url.txt` dengan tautan publiknya.
