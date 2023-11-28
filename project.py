import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('resource/Salary.csv')

x = df["YearsExperiece"].values
y = df["Salary"].values

x = np.array(x).reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

st.header('Pengaruh Pengalaman Kerja Terhadap Gaji', divider='rainbow')

with st.form("Penngalaman Kerja"):

    number = st.number_input("Pengalaman Kerja (Tahun)", min_value=0, placeholder="Masukkan angka...")

    submitted = st.form_submit_button("Submit")

if submitted:
    prediksi = regressor.predict(np.array([number]).reshape(-1, 1))
    st.info(f"Hasil Prediksi : ${prediksi[0]}")