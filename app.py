import streamlit as st
import yfinance as yf
import pandas as pd
import os

# --- KONFIGURASI DASAR & JUDUL ---
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📈 Dasbor Prediksi & Harga Real-time Bitcoin")
st.markdown("Dasbor ini menampilkan harga real-time dan prediksi harga Bitcoin untuk hari berikutnya menggunakan model TCN–BiLSTM–GRU.")

# --- FUNGSI-FUNGSI UTAMA ---
@st.cache_data(ttl=600) # Cache data selama 10 menit
def fetch_realtime_data():
    """Mengambil data harga 90 hari terakhir dari Yahoo Finance."""
    data = yf.download("BTC-USD", period="90d", interval="1d")
    data.index = data.index.date # Ubah indeks ke format tanggal saja
    return data

def load_prediction_history():
    """Memuat riwayat prediksi dari file CSV."""
    csv_file = 'prediction_history.csv'
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        return pd.DataFrame(columns=["Date", "Harga Prediksi", "Arah"])

# --- EKSEKUSI & TAMPILAN UI ---
try:
    df_realtime = fetch_realtime_data()
    df_history = load_prediction_history()

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
        st.dataframe(
            df_history.sort_values(by="Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

except Exception as e:
    st.error(f"Terjadi error: {e}")
