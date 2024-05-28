import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo_knn_entrenado = joblib.load('mohs_hardness_knn_modelo_entrenado.pkl')

# Crear la interfaz de usuario
st.title("PREDICCIÓN DE LA DUREZA DE MOH")
st.write('Ingrese los valores de las características para realizar una predicción:')

allElectronsTotal = st.number_input('allelectrons_Total')
densityTotal = st.number_input('density_Total')
atomicweightAverage = st.number_input('atomicweight_Average')

# Realizar la predicción con el modelo
input_data = np.array([[allElectronsTotal, densityTotal, atomicweightAverage]])
prediction = modelo_knn_entrenado.predict(input_data)

st.write('Este material tiene una dureza en la escala de Moh: ', prediction)

