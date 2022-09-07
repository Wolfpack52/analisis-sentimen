import streamlit as st
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set(font_scale=1.2)
from sklearn.svm import LinearSVC

n_bar = st.sidebar.radio("Navigation",["Home", "Results", "Visualization"])
#home section
if n_bar == "Home":
    st.markdown("<h1 style='text-align: center;'>Analisis Sentimen Aplikasi Bibit Menggunakan Metode SVM</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("logo_bibit.png")

    with col3:
        st.write(' ')
    
    st.markdown("<h2 style='text-align: center;'>Satria Pamungkas </h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>16118544 </h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Universitas Gunadarma </h2>", unsafe_allow_html=True)
    #st.subheader("Satria Pamungkas")
    #st.subheader("16118544")
    #st.subheader("4KA08")
    #st.subheader("Universitas Gunadarma")

#hasil
if n_bar == "Results":
    st.title("Results")

    df_hasil = pd.read_csv('hasil_prediksi_aktual.csv')
    df_hasil[['data_prediksi', 'hasil_prediksi', 'aktual']]

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "84%")
    col2.metric("System Failure", "16%")

    col3, col4 = st.columns(2)
    col3.metric("Positive Precision", "84%")
    col4.metric("Negative Precision", "83.98%")

    col5, col6 = st.columns(2)
    col5.metric("Positive Recall", "79.32%")
    col6.metric("Negative Recall", "87.78%")
    


#Visualisasi section
if n_bar == "Visualization":
    st.title("Visualization")

    st.subheader("Wordcloud: ")
    col1, col2 = st.columns(2)

    original = 'wordcloud_negatif.png'
    col1.header("Wordcloud Negatif")
    col1.image(original, use_column_width=True)

    grayscale = 'wordcloud_positif.png'
    col2.header("Wordcloud Positif")
    col2.image(grayscale, use_column_width=True)

    st.subheader("Chart: ")
    col1, col2 = st.columns(2)

    bar_chart = 'bar_chart.png'
    col1.header("Bar Chart")
    col1.image(bar_chart, use_column_width=True)

    pie_chart = 'pie_chart.png'
    col2.header("Pie Chart")
    col2.image(pie_chart, use_column_width=True)

    
