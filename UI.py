import streamlit as st
import time

from PIL import Image
#from prediksi import predict

import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
#from modelDens import model

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 7)


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

current_dir = os.getcwd()
PATH = os.path.join(current_dir, "modd.pth")

device = torch.device("cpu")
mod = model
mod.load_state_dict(torch.load(PATH, map_location=device))

# Make sure to call input = input.to(device) on any input tensors that you feed to the model

# mod= torch.load(PATH)
# mod.eval()
transfo =transforms.Compose([transforms.Resize(254),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.RandomHorizontalFlip(),
                             transforms.Normalize(mean, std)])

def predict(image_path):
    was_training = mod.training
    mod.eval()
    images_so_far = 0
    
    with torch.no_grad():
        img = Image.open(image_path)
        
        batch_t = torch.unsqueeze(transfo(img),0)
        batch_t = batch_t.to(device)
        
        outputs = mod(batch_t)
        print(outputs[0])
        _,preds = torch.max(outputs,1)
        
        mod.train(mode=was_training)
        
    return(preds)
   
def show(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.sidebar.markdown("<h2 style='text-align: center; color:black;'> Informasi Tambahan</h2>", unsafe_allow_html=True)
        st.sidebar.markdown("""
                            <p style='color:black;'>Contributors :</p>
                            <ol style='color: black;'>
                                <li> Dr.Eng.Fitra Abdurrachman Bachtiar, S.T., M.Eng. </li>
                                <li> Aldo Friska Darma Putra </li>
                            </ol>
                            """, unsafe_allow_html=True)
        st.sidebar.markdown("")
        st.sidebar.markdown("")
        st.sidebar.markdown("")
        st.sidebar.markdown("")
        st.sidebar.markdown("")
        st.sidebar.markdown("<h3 style'text-align: center; color:black'> Powered by</h3>",unsafe_allow_html=True)
        st.sidebar.markdown("""
                            <p align= "center"
                            <img src="D:\Smt7\Skripsi\demo\Logo_Universitas_Brawijaya.svg.png">
                            </p>
                            """, unsafe_allow_html= True)
        st.sidebar.markdown("<h4 style='text-align:center; color:black;'> Copyright 2022</h4>", unsafe_allow_html=True)
        st.sidebar.markdown("<h2 style='text-align:center;'> ISDDS: Facial Emotion Recognition</h2>",unsafe_allow_html=True)
        st.write("")
        st.write("""
                 <p>Rule :</p>
                 <ul>
                    <li> deteksi ini hanya digunakan untuk gambar emosi pada raut wajah</li>
                    <li> Akurasi dalam model ini adalah 88%</li>
                    <li> hasil output dalam program ini ada 7, yaitu: Angry, Disgust, Fear, Happy, Neutral, Sadness, dan Surprissed</li>
                    </ul>
                 """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        
        # Menambahkan gambar dari file lokal
        pa = os.path.join(current_dir, "sss.png")
        image = Image.open(pa)
        st.image(image, caption='Contoh Ekspresi')
        
        file_up = st.file_uploader("Upload an image")
        file_Cam = st.camera_input("take picture")
        
        if file_up is not None:
            image = Image.open(file_up)
            st.image(image, caption='Upload Image',use_column_width=True)
            classes =('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprissed')
            
            labels = predict(file_up)
            my_bar = st.progress(0)
            for percent_complete in range(70):
                time.sleep(0.001)
                my_bar.progress(percent_complete + 1)
            my_bar.empty()
            
            hasil = classes[labels]
            st.write('predicted', hasil)
            
        if file_Cam is not None:
            image = Image.open(file_Cam)
            st.image(image, caption='Upload Image',use_column_width=True)
            classes =('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprissed')
            
            labels = predict(file_Cam)
            my_bar = st.progress(0)
            for percent_complete in range(70):
                time.sleep(0.001)
                my_bar.progress(percent_complete + 1)
            my_bar.empty()
            
            hasil = classes[labels]
            st.write('predicted', hasil)
            
if __name__ == "__main__":
  show("style.css")
