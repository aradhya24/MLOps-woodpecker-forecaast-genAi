import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import holidays
from datetime import date, datetime, timedelta
import streamlit as st
import tensorflow as tf
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_option_menu import option_menu
# from keras.saving import register_keras_serializable

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.set_page_config(
        page_title="Demand Forecasting App",
        page_icon="📊"
)

import main
import image_bot
import data_bot


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = 0
      
        selected = option_menu(
        menu_title='',
        options=['Generate Forecasts','Chat with Image', 'Chat with Data'],
        icons=['cloud-arrow-up','graph-up-arrow', 'database-check'],
        menu_icon='chat-text-fill',
        default_index=st.session_state.selected_index,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "white"},
            "icon": {"color": "black", "font-size": "default"}, 
            "nav-link": {"color": "black", "font-size": "default", "text-align": "left", "margin": "0px", "--hover-color": "#e8f5e9"},
            "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
        }
        )

        st.session_state.selected_index = ['Generate Forecasts', 'Chat with Image', 'Chat with Data'].index(selected)

        
        if selected == "Generate Forecasts":
            main.app()
        if selected == "Chat with Image":
            image_bot.app()
        if selected == "Chat with Data":
            data_bot.app()
          
             
    run()      