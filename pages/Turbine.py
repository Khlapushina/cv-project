from ultralytics import YOLO
import streamlit as st
from PIL import Image
import io
import cv2
import os
import torch
import torchvision.transforms as transforms
import requests
from torchvision import transforms as T
import base64
import cv2
import numpy as np
import tempfile

@st.cache_resource

def load_model(weights_path):
    model = YOLO(weights_path)
    return model

model = load_model('weights/turbine.pt')
model = model.cpu()





image_type = st.radio("Способ загрузки", ["Олд", "Ньюфаг", "Тест"])

if image_type == "Олд":
    # image = st.file_uploader('Загрузи файл', type=['jpg', 'jpeg', 'png'])
    uploaded_images = st.file_uploader("Загрузите изображения", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded_images is not None:
        for image in uploaded_images:
            image = Image.open(image)
            # image_bytes = image.read()
            # image = Image.open(io.BytesIO(image_bytes))
            results = model.predict(image)
            result = results[0]
            img = Image.fromarray(result.plot()[:, :, ::-1])
            st.image(img)


if image_type == "Ньюфаг":
    image_url = st.text_input("Введите URL изображения для загрузки")
    if image_url:
        if image_url.startswith("data:image"):
            # Handle data URI
            image_data = image_url.split(',')[1]
            image_binary = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_binary))
        else:
            response = requests.get(image_url)
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
        results = model.predict(image)
        result = results[0]
        img = Image.fromarray(result.plot()[:, :, ::-1])
        st.image(img)
if image_type == "Тест":
    video_file = open('testvideo.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

