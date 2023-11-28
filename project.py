import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('resource/Salary.csv')

x = df["YearsExperience"].values
y = df["Salary"].values

x = np.array(x).reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

st.header('Pengaruh Pengalaman Kerja Terhadap Gaji', divider='rainbow')
st.subheader('Prediksi gaji per bulan untuk karyawan di Amerika Serikat')

with st.form("Penngalaman Kerja"):

    number = st.number_input("Pengalaman Kerja (Tahun)", format="%0.5g", min_value=0.0, max_value=50.0, step=0.1, placeholder="Masukkan angka...")

    submitted = st.form_submit_button("Submit")

if submitted:
    prediksi = regressor.predict(np.array([number]).reshape(-1, 1))

    prediksi_value = prediksi[0]
    hasil = f"Hasil Prediksi: ${'{:,.2f}'.format(prediksi_value)}"
    st.info(hasil)

