import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tcn import TCN

# --- Konfigurasi Dasar Aplikasi ---
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📊 Prediksi Harga Bitcoin Otomatis")
st.markdown("Menampilkan harga terbaru dan prediksi untuk hari berikutnya menggunakan model TCN–BiLSTM–GRU.")

# --- Fungsi-fungsi Utama ---
@st.cache_resource
def load_model_and_scaler():
    """Memuat model dan scaler yang sudah di-fine-tune."""
    try:
        model = load_model("model_finetuned_btc_2025.h5", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scaler_finetuned.save")
        return model, scaler
    except FileNotFoundError:
        # Jika file tidak ditemukan, kembalikan None agar bisa ditangani
        return None, None

@st.cache_data(ttl=600) # Simpan cache data selama 10 menit
def fetch_btc_data(window_size=60):
    """Mengambil data harga 60 hari terakhir dari Yahoo Finance."""
    # Ambil data 90 hari untuk memastikan tidak ada data yang hilang
    df = yf.download("BTC-USD", period="90d", interval="1d")
    df = df[["Close"]].dropna()
    return df.tail(window_size)

# --- Logika Utama Aplikasi ---
model, scaler = load_model_and_scaler()

# Tampilkan pesan error jika model belum ada
if model is None or scaler is None:
    st.error(
        "❌ Model atau scaler tidak ditemukan. "
        "Pastikan workflow 'Daily Model Finetuning' di GitHub Actions sudah berhasil dijalankan setidaknya satu kali."
    )
else:
    try:
        # 1. Ambil data
        df = fetch_btc_data(window_size=60)

        # 2. Preprocess dan Prediksi
        scaled_data = scaler.transform(df)
        X_input = np.array([scaled_data]) # Reshape untuk input model
        pred_scaled = model.predict(X_input)
        predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]

        # 3. Tampilkan UI
        st.subheader("📅 Tabel Harga Bitcoin (60 Hari Terakhir)")
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

        st.divider()

        st.subheader("🔮 Hasil Prediksi Harga Berikutnya")
        
        # Tambahkan tanggal prediksi untuk konteks
        prediction_date = (pd.to_datetime("today") + pd.DateOffset(days=1)).date()
        harga_terakhir = df['Close'].iloc[-1]
        
        col1, col2 = st.columns([1, 2]) # Buat kolom agar lebih rapi
        with col1:
            st.metric(
                label=f"Prediksi untuk {prediction_date.strftime('%d %B %Y')}",
                value=f"${predicted_price:,.2f}",
                delta=f"${predicted_price - harga_terakhir:,.2f}"
            )
        with col2:
            st.info("ℹ️ Model ini diperbarui setiap hari melalui GitHub Actions untuk menjaga akurasi. Prediksi di atas adalah untuk harga penutupan hari berikutnya.")

    except Exception as e:
        st.error(f"Terjadi error saat memproses data atau membuat prediksi: {e}")
