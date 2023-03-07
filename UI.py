import streamlit as st
import time

from PIL import Image
from prediksi import predict

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
    #st.beta_set_page_config(page_title='Facial emotion Recognition', page_icon='https://www.google.com/url?sa=i&url=https%3A%2F%2Fid.m.wikipedia.org%2Fwiki%2FBerkas%3ALogo_Universitas_Brawijaya.svg&psig=AOvVaw0y34shl4uYh5IA6ziBsXyq&ust=1672144783278000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCOi48fCml_wCFQAAAAAdAAAAABAE')
    show("style.css")
        
        
        
        
