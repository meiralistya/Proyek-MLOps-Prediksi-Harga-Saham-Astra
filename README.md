# 📈 Proyek MLOps: Prediksi Harga Saham Astra (ASII.JK)

Selamat datang di **Proyek Prediksi Harga Saham Astra (ASII.JK)**.
Proyek ini bertujuan untuk membangun **pipeline MLOps end-to-end** untuk memprediksi harga saham menggunakan data historis.

🔹 Fokus utama proyek ini adalah **implementasi workflow MLOps**, meliputi deployment, CI/CD, dan monitoring, **bukan kompleksitas model machine learning**.

---

## 🎯 Tujuan Proyek

* Menerapkan alur kerja MLOps secara end-to-end
* Mengelola data, model, dan eksperimen secara terstruktur
* Menyediakan model dalam bentuk **REST API**
* Menerapkan **CI/CD menggunakan GitHub Actions**
* Menerapkan **monitoring dasar melalui logging**

---

## 🏗️ Arsitektur Sistem

```
Data Historis Saham
        ↓
Preprocessing Data
        ↓
Training Model & Tracking (MLflow)
        ↓
Model Terbaik (best_model.pkl)
        ↓
API Inference (FastAPI)
        ↓
Cloud Deployment
        ↓
CI/CD (GitHub Actions)
        ↓
Monitoring & Logging
```

---

## 📂 Struktur Repository

```
.
├── data/
├── models/
│   └── best_model.pkl
├── src/
│   └── serving/
│       └── app.py
├── predictions/
├── train.py
├── tune.py
├── predict.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 🚀 Menjalankan Aplikasi

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<username>/mlops-astra-stock-prediction.git
cd mlops-astra-stock-prediction
```

---

### 2️⃣ Install Dependency

Pastikan Python 3.8+ telah terinstal.

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Menjalankan API FastAPI

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
```

Jika berhasil, API dapat diakses di:

```http
http://localhost:8080
```

---

### 4️⃣ Endpoint Prediksi

```http
POST /predict
```

Contoh request:

```json
{
  "open": 8000,
  "high": 8200,
  "low": 7900,
  "volume": 1000000
}
```

Contoh response:

```json
{
  "prediction": [8150.32]
}
```

---

## 🐳 Docker

Aplikasi dikemas menggunakan Docker untuk memastikan konsistensi environment.

### Build Image

```bash
docker build -t mlops-astra .
```

### Run Container

```bash
docker run -p 8080:8080 mlops-astra
```

---

## 🔁 CI/CD Pipeline

Proyek ini menggunakan **GitHub Actions** untuk mengotomatisasi proses Continuous Integration.

Pipeline dijalankan setiap kali terjadi **push ke repository** dan mencakup tahapan:

* Instalasi dependency
* Pengujian dasar aplikasi FastAPI
* Build Docker image

CI/CD memastikan aplikasi selalu berada dalam kondisi siap untuk deployment.

---

## 📊 Monitoring

Monitoring dilakukan melalui **logging pada API FastAPI**, meliputi:

* Pencatatan request yang masuk
* Logging hasil prediksi
* Logging error aplikasi

Monitoring ini digunakan untuk memantau aktivitas sistem dan mendeteksi error secara dini.
Pengembangan lanjutan dapat mencakup integrasi tools seperti Prometheus dan Grafana.

---

## 👥 Pembagian Tugas Tim

| Nama                   | NIM       | Peran          | Tanggung Jawab                   |
| ---------------------- | --------- | -------------- | -------------------------------- |
| Salwa Farhanatussaidah | 122450011 | Data Engineer  | Data ingestion, preprocessing    |
| Tria Yunanni           | 122450062 | ML Engineer    | Training model, evaluasi, MLflow |
| Meira Listyaningrum    | 122450055 | MLOps Engineer | API, Docker, deployment          |
| Chalifia Wananda       | 122450076 | DevOps / PM    | CI/CD, monitoring, dokumentasi   |

---

## 📜 Lisensi

Proyek ini menggunakan **MIT License**.

---
