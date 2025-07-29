import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tcn import TCN
import os

# Fungsi ini khusus untuk membuat prediksi dan memperbarui file CSV
def update_prediction_file():
    # Muat model dan scaler yang baru di-fine-tune
    model = load_model("model_finetuned_btc_2025.h5", compile=False, custom_objects={"TCN": TCN})
    scaler = joblib.load("scaler_finetuned.save")

    # Ambil 60 hari data terakhir untuk membuat input
    df_for_input = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1640995200&period2=4102444800&interval=1d&events=history&includeAdjustedClose=true")
    df_for_input = df_for_input[['Close']].tail(60)

    # Buat prediksi
    scaled_data = scaler.transform(df_for_input)
    X_input = np.array([scaled_data]).reshape((1, 60, 1))
    pred_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]

    # Siapkan data baru untuk disimpan
    tomorrow_date = pd.to_datetime("today").date() + pd.DateOffset(days=1).date()
    harga_aktual_terakhir = df_for_input['Close'].iloc[-1]
    arah = "Naik 📈" if predicted_price > harga_aktual_terakhir else "Turun 📉"

    new_prediction = pd.DataFrame([{
        "Date": tomorrow_date,
        "Harga Prediksi": f"${predicted_price:,.2f}", # Format sebagai string
        "Arah": arah
    }])

    # Muat, perbarui, dan simpan riwayat prediksi
    csv_file = 'prediction_history.csv'
    if os.path.exists(csv_file):
        history_df = pd.read_csv(csv_file)
    else:
        history_df = pd.DataFrame(columns=["Date", "Harga Prediksi", "Arah"])

    history_df['Date'] = pd.to_datetime(history_df['Date']).dt.date
    history_df = history_df[history_df['Date'] != tomorrow_date]
    history_df = pd.concat([history_df, new_prediction], ignore_index=True)

    history_df.to_csv(csv_file, index=False)
    print(f"Successfully updated {csv_file} for date {tomorrow_date}")

# Jalankan fungsi utama
if __name__ == "__main__":
    update_prediction_file()
