import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tcn import TCN

# Konfigurasi dasar
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="centered")
st.title("📈 Prediksi Harga Bitcoin")

try:
    # Muat model dan scaler
    @st.cache_resource
    def load_prediction_model():
        model = load_model("model_finetuned_btc_2025.h5", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scaler_finetuned.save")
        return model, scaler

    model, scaler = load_prediction_model()

    # Ambil data 60 hari terakhir untuk input
    df_input = yf.download("BTC-USD", period="90d", interval="1d")[['Close']].tail(60)

    # Preprocess dan buat prediksi
    scaled_data = scaler.transform(df_input)
    X_input = np.array([scaled_data]).reshape((1, 60, 1))
    pred_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]
    
    # Tampilkan hasil
    prediction_date = pd.to_datetime("today").date() + pd.DateOffset(days=1).date()
    harga_kemarin = df_input['Close'].iloc[-1]

    st.success(f"Prediksi berhasil dibuat!")
    st.metric(
        label=f"Prediksi Harga BTC untuk {prediction_date.strftime('%d %B %Y')}",
        value=f"${predicted_price:,.2f}",
        delta=f"${predicted_price - harga_kemarin:,.2f} vs kemarin"
    )
    st.info("Model diperbarui setiap hari untuk akurasi yang lebih baik.")

except FileNotFoundError:
    st.error("Model (`model_finetuned_btc_2025.h5`) belum tersedia. Harap jalankan workflow di GitHub Actions terlebih dahulu.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
