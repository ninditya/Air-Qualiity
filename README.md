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

Empat pertanyaan analisis yang dijawab (sesuai kriteria minimum: 2 bisnis + 2 analisis lanjutan tanpa ML):
1. Q1 (Pertanyaan Bisnis): Area mana yang memerlukan intervensi prioritas berdasarkan tingkat polusi PM2.5 dan gas polutan, serta bagaimana tren perbaikannya dari waktu ke waktu?
2. Q2 (Pertanyaan Bisnis): Kapan waktu-waktu kritis yang perlu menjadi fokus pengawasan emisi dan perlindungan kesehatan masyarakat?
3. Q3 (Pertanyaan Analisis Lanjutan tanpa ML): Faktor meteorologi apa yang paling berkontribusi terhadap peningkatan risiko polusi, sehingga bisa dijadikan indikator peringatan dini?
4. Q4 (Pertanyaan Analisis Lanjutan - AQI Multi-Polutan tanpa ML): Polutan apa yang menjadi penyebab utama penurunan kualitas udara, dan bagaimana prioritas pengendaliannya berdasarkan tingkat risikonya?

## Struktur Direktori

```text
submission-final/
├── dashboard/
│   ├── dashboard.py
│   └── main_data.csv
├── data/
│   ├── air_quality_hourly_all_stations.csv
│   ├── station_coordinates.csv
│   └── raw/
├── notebook.ipynb
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
jupyter notebook notebook.ipynb
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

Dashboard menampilkan ringkasan metrik di bagian atas dan terdiri dari 4 tab utama:
1. **Area Prioritas (Q1)**: tren PM2.5, ranking stasiun prioritas, perbandingan zona geospasial, dan clustering manual stasiun berbasis gas polutan.
2. **Waktu Kritis (Q2)**: tren bulanan, heatmap jam-bulan, distribusi musiman, serta ringkasan bulan/jam puncak polutan.
3. **Dampak Cuaca (Q3)**: korelasi Spearman PM-meteorologi dan gas-meteorologi pada skala musim/tahun/bulan/jam, plus analisis arah angin (`wd`).
4. **Risiko AQI (Q4)**: distribusi kategori AQI overall resmi, ranking polutan berisiko, dan ringkasan stasiun terbaik/terburuk berdasarkan AQI.

## Catatan

- File `dashboard/main_data.csv` diekspor dari notebook agar analisis notebook dan dashboard konsisten.
- Koordinat stasiun dipakai sebagai proxy visual geospasial untuk perbandingan relatif antarzona.
- Jika dashboard sudah dideploy ke Streamlit Community Cloud, isi `url.txt` dengan tautan publiknya.
