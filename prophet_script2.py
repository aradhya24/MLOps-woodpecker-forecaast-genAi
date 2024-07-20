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

model = Prophet()


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

# Define functions specific to the Prophet model
def read_process(file):
    df = pd.read_csv(file)
    
    # Try to parse the 'Date' column with multiple formats
    date_formats = ["%d-%m-%Y", "%m/%d/%Y"]
    for date_format in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format)
            break
        except ValueError:
            continue
    
    # If the date parsing failed, raise an error
    if df['Date'].isna().any():
        raise ValueError("Date parsing failed for all formats")
    
    # Convert the 'Date' column to the desired format
    df['Date'] = df['Date'].dt.strftime("%m/%d/%Y")
    
    # Create the new DataFrame with the required columns
    data = pd.DataFrame()
    data["ds"] = df["Date"]
    data["y"] = df["Sales"]
    
    return data

def evaluate(df, end_date):
    df['ds'] = pd.to_datetime(df['ds'])
    start_date = df["ds"].iloc[-1]
    d_1 = df["ds"].iloc[0]
    d_2 = df["ds"].iloc[1]
    if isinstance(end_date, date):
        end_date = pd.Timestamp(end_date)
    diff_dates = (d_2 - d_1).days
    if diff_dates == 1:
        days_selected = (end_date - start_date).days
        st.write(f"Number of Days selected: {days_selected}")
        return days_selected, diff_dates
    
    if diff_dates == 7:
        weeks_selected = (end_date - start_date).days // 7
        st.write(f"Number of Weeks selected: {weeks_selected}")
        return weeks_selected, diff_dates
    return days_selected, diff_dates

def forecast(model, df, timesteps, f):
    model.fit(df)
    if f == 1:
        future_df = model.make_future_dataframe(periods=timesteps, freq='D')
    if f == 7:
        future_df = model.make_future_dataframe(periods=timesteps, freq='W')
    future_forecasts = model.predict(future_df)
    future_forecasts = model.predict(future_df)
    dates = future_forecasts["ds"]
    preds = future_forecasts["yhat"]
    last_idx = df.index[-1]
    fig = plot_time_series(timesteps=dates[last_idx:], values=preds[last_idx:])
    return future_forecasts, last_idx, fig
