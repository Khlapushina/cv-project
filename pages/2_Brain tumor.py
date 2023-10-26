import streamlit as st
from PIL import Image
import torch
from torchvision import io
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from torchvision import transforms as T
model = torch.hub.load(
    'yolov5/', # пути будем указывать гдето в локальном пространстве
    'custom', # непредобученная
    path='braintumor.pt', # путь к нашим весам
    source='local' # откуда берем модель – наша локальная
    )


uploaded_file = st.file_uploader("Загрузите фотографию", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    if st.button("Определить наличие опухоли"):
        image.save("ggg.jpg")

        model.conf = 0.3 # устанавливаем порог
        # Image
        img = 'ggg.jpg'
        model.eval()
        # Inference
        results = model(img)
        # results.show()  # or .show(), .save(), .crop(), .pandas(), etc
        annotated_img = results.render()[0]
        st.image(annotated_img, caption='Результат', use_column_width=True)