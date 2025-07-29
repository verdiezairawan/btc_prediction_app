import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tcn import TCN
import os

# --- KONFIGURASI DASAR & JUDUL ---
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📈 Dasbor Prediksi & Harga Real-time Bitcoin")
st.markdown("Dasbor ini menampilkan harga real-time dan prediksi harga Bitcoin untuk hari berikutnya menggunakan model TCN–BiLSTM–GRU.")

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_resource
def load_prediction_model():
    """Memuat model dan scaler yang sudah di-fine-tune."""
    model = load_model("model_finetuned_btc_2025.h5", compile=False, custom_objects={"TCN": TCN})
    scaler = joblib.load("scaler_finetuned.save")
    return model, scaler

@st.cache_data(ttl=600) # Cache data selama 10 menit
def fetch_realtime_data():
    """Mengambil data harga 90 hari terakhir dari Yahoo Finance."""
    data = yf.download("BTC-USD", period="90d", interval="1d")
    data.index = data.index.date
    return data

def update_and_get_predictions_history():
    """
    Membuat prediksi baru, lalu memuat, memperbarui, dan menyimpan
    riwayat prediksi dalam file CSV.
    """
    # 1. Buat prediksi untuk besok
    df_for_input = df_realtime[['Close']].tail(60)
    scaled_data = scaler.transform(df_for_input)
    X_input = np.array([scaled_data]).reshape((1, 60, 1))
    pred_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]

    # 2. Tentukan tanggal prediksi (besok) dan siapkan data baru
    tomorrow_date = pd.to_datetime("today").date() + pd.DateOffset(days=1).date()
    harga_aktual_terakhir = df_realtime['Close'].iloc[-1]
    arah = "Naik 📈" if predicted_price > harga_aktual_terakhir else "Turun 📉"
    
    new_prediction = pd.DataFrame([{
        "Date": tomorrow_date,
        "Harga Prediksi": predicted_price,
        "Arah": arah
    }])

    # 3. Muat, perbarui, dan simpan riwayat prediksi
    csv_file = 'prediction_history.csv'
    if os.path.exists(csv_file):
        history_df = pd.read_csv(csv_file)
    else:
        history_df = pd.DataFrame(columns=["Date", "Harga Prediksi", "Arah"])

    # Pastikan tipe data tanggal konsisten sebelum merge
    history_df['Date'] = pd.to_datetime(history_df['Date']).dt.date
    
    # Hapus prediksi lama untuk tanggal yang sama (jika ada), lalu tambahkan yang baru
    history_df = history_df[history_df['Date'] != tomorrow_date]
    history_df = pd.concat([history_df, new_prediction], ignore_index=True)
    
    # Simpan kembali ke CSV
    history_df.to_csv(csv_file, index=False)
    
    return history_df

# --- EKSEKUSI APLIKASI ---

# Muat model, scaler, dan data
try:
    model, scaler = load_prediction_model()
    df_realtime = fetch_realtime_data()
    
    # Update dan dapatkan riwayat prediksi
    df_history = update_and_get_predictions_history()

    # --- TAMPILAN UI ---

    # 1. Grafik Harga Real-time
    st.subheader("Grafik Harga Real-time (90 Hari Terakhir)")
    st.line_chart(df_realtime['Close'])

    # Gunakan kolom untuk tata letak yang lebih rapi
    col1, col2 = st.columns(2)

    with col1:
        # 2. Tabel Harga Real-time
        st.subheader("Tabel Harga Real-time")
        st.dataframe(
            df_realtime.sort_index(ascending=False),
            use_container_width=True
        )

    with col2:
        # 3. Tabel Hasil Prediksi
        st.subheader("Tabel Riwayat Prediksi")
        df_history_display = df_history.copy()
        df_history_display['Date'] = pd.to_datetime(df_history_display['Date']).dt.strftime('%Y-%m-%d')
        st.dataframe(
            df_history_display.sort_values(by="Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

except FileNotFoundError:
    st.error(
        "Model atau scaler belum tersedia. Harap jalankan workflow 'Daily Model Finetuning' di GitHub Actions terlebih dahulu."
    )
except Exception as e:
    st.error(f"Terjadi error: {e}")
