import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if "conversation_history_data" not in st.session_state:
    st.session_state.conversation_history_data = []

def add_custom_css_data_bot():
    st.markdown("""
    <style>
    .user-message {
        background-color: #e0f7fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #f1f8e9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def app():
    add_custom_css_data_bot()
    st.title("Data Analysis and Context Generation")

    # Upload and display the data file
    uploaded_file = st.file_uploader("Choose a data file...", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Read the data file
        if uploaded_file.type == "text/csv":
            import pandas as pd
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            import pandas as pd
            data = pd.read_excel(uploaded_file)
        
        st.write(data)

        # Add chat message for the initial prompt
        st.chat_message("📊").write("Analyze the trends in this dataset.")

        user_prompt = st.chat_input("Enter your prompt here:")
        
        if user_prompt:
            # Hardcoded default prompt for analyzing datasets
            default_prompt = """
                The model will be getting a CSV file containing columns of dates and forecast values of sales or prices.
                The model has to understand the data thoroughly and generate responses.
            """
            
            # Combine the default prompt with the user-provided prompt
            combined_prompt = f"{default_prompt}\n{user_prompt}"

            # Convert the data to text for the model (could be JSON, CSV, etc.)
            data_text = data.to_string()

            # Pass the combined prompt and data text to the model
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                [combined_prompt, data_text],
                generation_config = genai.types.GenerationConfig(
                                  top_p = 0.6,
                                  top_k = 5,
                                  temperature = 0.8),
                stream=True)
            response.resolve()

            st.session_state.conversation_history_data.append(("👦🏻", user_prompt, "user-message"))
            st.session_state.conversation_history_data.append(("🤖", response.text, "bot-message"))

            # Display the conversation history
            for speaker, message, css_class in st.session_state.conversation_history_data:
                st.markdown(f'<div class="{css_class}">{speaker} : {message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
