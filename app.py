import streamlit as st
import requests
from PIL import Image
import io

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to predict if it shows Pneumonia or is Normal.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Chest X-ray.', use_column_width=True)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    
    response = requests.post("http://localhost:8000/predict/", files={"file": ("image.jpg", img_byte_arr, "image/jpeg")})
    
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: {result['prediction']}")
    else:
        st.write("Error in prediction")
