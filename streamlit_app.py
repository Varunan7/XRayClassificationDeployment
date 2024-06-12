import streamlit as st
import requests
from PIL import Image

st.title('X-Ray Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button('Classify'):
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            files={"file": uploaded_file.getvalue()}
        )

        if response.status_code == 200:
            result = response.json()
            st.write(f"CNN Prediction: {result['cnn_prediction']}")
            st.write(f"DNN Prediction: {result['dnn_prediction']}")
        else:
            st.write("Error: ", response.text)
