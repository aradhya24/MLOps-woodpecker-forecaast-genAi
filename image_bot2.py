import google.generativeai as genai
import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


if "conversation_history_image" not in st.session_state:
    st.session_state.conversation_history_image = []


def add_custom_css_image_bot():
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
    add_custom_css_image_bot()
    st.title("Image Description and Context Generation")

    # Load and display the image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Convert the image to bytes
        # img_byte_arr = io.BytesIO()
        # image.save(img_byte_arr, format=image.format)
        # img_byte_arr = img_byte_arr.getvalue()

        # Add chat message for the initial prompt
        st.chat_message("📈").write("Analyze the trends in this graph.")

        user_prompt = st.chat_input("Enter your prompt here:")
        
        if user_prompt:
            # Hardcoded default prompt for forecasting graphs and trend analysis
            default_prompt = """
                You are an expert in analyzing forecasting graphs for trend analysis.
                You will receive input images as graphs and you will have to answer questions based on the observed trends in brief and elaborate it.
            """
            
            # Combine the default prompt with the user-provided prompt
            combined_prompt = f"{default_prompt}\n{user_prompt}"

            # Pass the combined prompt and image bytes to the model
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                [combined_prompt, image], 
                generation_config = genai.types.GenerationConfig(
                                  temperature = 1.0),
                stream=True)
            response.resolve()

            st.session_state.conversation_history_image.append(("👦🏻", user_prompt, "user-message"))
            st.session_state.conversation_history_image.append(("🤖", response.text, "bot-message"))

            # Display the conversation history
            for speaker, message, css_class in st.session_state.conversation_history_image:
                st.markdown(f'<div class="{css_class}">{speaker} : {message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
