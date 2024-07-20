import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from datetime import datetime
import tensorflow as tf
from io import BytesIO
import os
from dotenv import load_dotenv
import time


from prophet_script2 import read_process, evaluate, forecast
from nbeats2 import read_and_process_nbeats, make_forecast_dates, make_future_forecast, NBeatsBlock, plot_time_series, WINDOW_SIZE



load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Helper function to save Matplotlib figure to bytes
def save_fig_to_bytes(fig):
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return img_bytes

# Functions to generate forecasts and files
def generate_prophet_files(uploaded_file, end_date):
    model = Prophet()
    df = read_process(uploaded_file)
    timesteps, freq = evaluate(df, end_date)
    fut, last_idx, fig = forecast(model, df, timesteps, freq)
    download_file = pd.DataFrame()
    download_file["Date"] = fut["ds"][last_idx:]
    download_file["Predictions"] = fut["yhat"][last_idx:]
    download_file = download_file.reset_index(drop=True)
    csv = download_file.to_csv(index=False).encode('utf-8')
    img_bytes = save_fig_to_bytes(fig)
    return csv, img_bytes

def generate_nbeats_files(uploaded_file, end_date):
    nbeats_model = tf.keras.models.load_model("C:/Users/Siddharth/Desktop/woodpeckers/nbeats.keras", custom_objects={'NBeatsBlock': NBeatsBlock})
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df['Date'] = df['Date'].dt.strftime("%m/%d/%Y")
    a, b = read_and_process_nbeats(df)
    x, y = make_forecast_dates(df, end_date)
    preds = make_future_forecast(b, nbeats_model, y, WINDOW_SIZE)
    forecast_df = pd.DataFrame()
    forecast_df["Date"] = x
    forecast_df["Predictions"] = preds
    forecast_df = forecast_df.reset_index(drop=True)
    fig = plot_time_series(timesteps=forecast_df["Date"], values=forecast_df["Predictions"])
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    img_bytes = save_fig_to_bytes(fig)
    return csv, img_bytes

# Main application function
def app():
    st.title("Generate Forecasts")
    st.header("File Upload")

    uploaded_file = st.file_uploader("Upload a File", type="csv")
    end_date = st.date_input("Enter Last Date to be Forecasted", datetime(2019, 7, 6))

    model_selection = st.selectbox("Select Model to generate Forecasts", ["Prophet", "N-Beats"], index=None, placeholder="Models..")

    # Unique keys with timestamps to ensure no conflicts
    timestamp = int(time.time())
    csv_key = f'csv_download_button_{model_selection}_{timestamp}'
    img_key = f'img_download_button_{model_selection}_{timestamp}'

    # Generate forecasts and files only if button is clicked
    if st.button("Generate Forecasts?"):
        if uploaded_file is not None:
            if model_selection == "Prophet":
                st.session_state.csv_data, st.session_state.img_data = generate_prophet_files(uploaded_file, end_date)
                st.success("Forecasts Generated")
            elif model_selection == "N-Beats":
                st.session_state.csv_data, st.session_state.img_data = generate_nbeats_files(uploaded_file, end_date)
                st.success("Forecasts Generated")

    # Display download buttons if files are generated
    if 'csv_data' in st.session_state and 'img_data' in st.session_state:
        st.download_button(
            label="Download Forecasts as CSV",
            data=st.session_state.csv_data,
            file_name='forecasts.csv',
            mime='text/csv',
            key=csv_key
        )
        st.download_button(
            label="Download Forecast as Image",
            data=st.session_state.img_data,
            file_name='forecast.png',
            mime='image/png',
            key=img_key
        )

if __name__ == "__main__":
    app()