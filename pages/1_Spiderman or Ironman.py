import base64
import streamlit as st
from PIL import Image
import torch
from torchvision import io
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from torchvision import transforms as T
model = torch.hub.load(
    'yolov5/', # пути будем указывать гдето в локальном пространстве
    'custom', # непредобученная
    path='weights/spidyandiron.pt', # путь к нашим весам
    source='local' # откуда берем модель – наша локальная
    )

st.title("Single Detection")
uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

model.eval()
model.conf = 0.3 

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image', use_column_width=True)

    if st.button("Detection (Single)"):
        image.save("ggg.jpg")

        # Image
        img = 'ggg.jpg'
        # Inference
        results = model(img)
        # results.show()  # or .show(), .save(), .crop(), .pandas(), etc
        annotated_img = results.render()[0]
        st.image(annotated_img, caption='Detection Result', use_column_width=True)
    

st.title("Multiple Detection")
uploaded_files = st.file_uploader("Upload your images", accept_multiple_files=True)

images = []

if uploaded_files is not None:
    for image in uploaded_files:
        # Преобразуйте файл изображения в объект PIL
        image = Image.open(image)
        st.image(image, caption='Original Image', use_column_width=True)
        images.append(image)

if st.button("Detection (Multiple)"):
    # Примените модель YOLOv5 к изображению
    for image in images:

        results = model(image)

        # Отобразите результаты
        st.image(results.render()[0], caption='Detection Result', use_column_width=True)


st.title("Upload image by URL")
    
# Ввод URL изображения
url = st.text_input("Enter image URL:")


    
# При нажатии на кнопку "Загрузить", выводим изображение
if st.button("Upload"):
    try:
        if url.startswith("data:image"):
            # Handle data URI
            image_data = url.split(',')[1]
            image_binary = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_binary))
        else:
            response = requests.get(url)
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))

        st.image(image, caption='Uploaded image', use_column_width=True)

        result = model(image)

        # Отобразите результаты
        st.image(result.render()[0], caption='Detection Result', use_column_width=True)
        
    except Exception as e:
        st.error("Error: " + str(e))

#if st.button("Detection (URL)"):

    # Inference
    #result = model(image)

    # Отобразите результаты
    #st.image(result.render()[0], caption='Detection Result', use_column_width=True)