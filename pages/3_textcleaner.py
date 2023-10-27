import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import torch
from PIL import Image
from weights.model import  ConvAutoencoder
from weights.preprocessing import preprocess
from torchvision.io import read_image

@st.cache_resource

def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('weights/autoencoder.pt', map_location = 'cpu'))
    return model

DEVICE = 'cpu'
model = load_model()
model.to(DEVICE)
model.eval()

loaded_image = st.file_uploader('Загрузите картинку с текстом')

def predict(img):
    img = preprocess(img)
    img.to(DEVICE)
    outputs = model(img.unsqueeze(0))
    pred = outputs.detach().cpu().squeeze(0).numpy()
    return pred

if loaded_image:
    img = Image.open(loaded_image)
    prediction = predict(img)
    left_col, right_col = st.columns(2)
    with left_col:
        st.write('Original text')
        st.image(img)
    with right_col:
        st.write('Denoised text')
        st.image(prediction[0])


