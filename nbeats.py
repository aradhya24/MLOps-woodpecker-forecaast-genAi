import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import holidays
from datetime import date, datetime, timedelta
import streamlit as st
import tensorflow as tf
# from keras.saving import register_keras_serializable

def plot_time_series(timesteps, values, format='-', start=0, end=None, label=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(timesteps[start:end], values[start:end], format, label=label)
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Forecasted Values of Sales")
    if label:
        ax.legend(fontsize=10)
    ax.grid(True)
    st.pyplot(fig)
    return fig




WINDOW_SIZE = 7
HORIZON = 1

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation='linear', name='theta')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast
    


def read_and_process_nbeats(df, window_size):
    date_formats = [
        "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", 
        "%Y/%m/%d", "%b %d, %Y", "%d %b %Y", "%d %B %Y"
    ]
    
    df['ParsedDate'] = pd.NaT
    for date_format in date_formats:
        try:
            df['ParsedDate'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
            if df['ParsedDate'].notna().all():
                break
        except ValueError:
            continue

    # If the date parsing failed, raise an error with detailed info
    if df['ParsedDate'].isna().any():
        unparsable_dates = df[df['ParsedDate'].isna()]['Date']
        raise ValueError(f"Date parsing failed for all formats. Unparsable dates: {unparsable_dates.tolist()}")
    
    df['Date'] = df['ParsedDate']
    df["Date"] = pd.to_datetime(df["Date"])
    df.drop(columns=['ParsedDate'], inplace=True)
    
    data = pd.DataFrame()
    data["ds"] = df["Date"]
    data["y"] = df["Sales"]
    data = data.set_index("ds")
    data_nbeats = data.copy()
    
    for i in range(window_size):
        data_nbeats[f"y + {i+1}"] = data_nbeats["y"].shift(periods=i+1)
    
    X_all = data_nbeats.dropna().drop("y", axis=1)
    y_all = data_nbeats.dropna()["y"]
    
    return X_all, y_all


def make_forecast_dates_daily(df, end_date):
    start_date = df.iloc[-1]["Date"]
    dates_to_be_forecasted = pd.date_range(start=start_date, end=end_date, freq='D')
    dates_to_be_forecasted = dates_to_be_forecasted[1:]
    st.write(f"Number of Timesteps selected: {len(dates_to_be_forecasted)}")
    return dates_to_be_forecasted, len(dates_to_be_forecasted)


def make_forecast_dates_weekly(df, end_date):
    start_date = df.iloc[-1]["Date"]
    dates_to_be_forecasted = pd.date_range(start=start_date, end=end_date, freq='W')
    dates_to_be_forecasted = dates_to_be_forecasted[1:]
    st.write(f"Number of Timesteps selected: {len(dates_to_be_forecasted)}")
    return dates_to_be_forecasted, len(dates_to_be_forecasted)

def make_future_forecast(values, model, into_future, window_size=WINDOW_SIZE) -> list:
    future_forecast = []
    last_window = values[-WINDOW_SIZE:]
    last_window = np.asarray(last_window)
    for _ in range(into_future):
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on:\n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")
        future_forecast.append(tf.squeeze(future_pred).numpy())
        last_window = np.append(last_window, future_pred)[-WINDOW_SIZE:]
    return future_forecast