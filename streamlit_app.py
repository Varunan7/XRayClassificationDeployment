import streamlit as st
import requests
from PIL import Image

st.title("X-Ray Image Classification")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    model_type = st.selectbox('Choose Model', ['cnn', 'dnn'])

    files = {'image': uploaded_file.getvalue()}
    data = {'model': model_type}
    response = requests.post("http://localhost:5000/predict", files=files, data=data)

    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Error in prediction")

