import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('resource/StressLevelDataset.csv')

x = df["sleep_quality"].values
y = df["stress_level"].values

x = np.array(x).reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

st.header('Pengaruh Kualitas Tidur Terhadap Tingkat Stress Mahasiswa', divider='rainbow')
st.subheader('Kualitas tidur adalah seberapa baik pola tidurmu dan seberapa banyak waktu tidurmu.')

with st.form("Kualitas Tidur"):

    number = st.number_input("Kualitas Tidur (0 = Kualitas tidur buruk, 5 = Kualitas tidur baik)", max_value=5, min_value=0, placeholder="Masukkan angka antara 0 - 5")

    submitted = st.form_submit_button("Submit")

if submitted:
    prediksi = regressor.predict(np.array([number]).reshape(-1, 1))
    st.info(f"Hasil Prediksi : {prediksi[0]}")