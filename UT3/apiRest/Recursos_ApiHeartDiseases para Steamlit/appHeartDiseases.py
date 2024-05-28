import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import streamlit as st
import joblib


# Path del modelo preentrenado
MODEL_PATH = 'models/Ejemp_3_3_modelo_entrenado_joblib.pkl'



# Se reciben los valores y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        model = joblib.load(MODEL_PATH)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Estudio de enfermedades cardíacas</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lectura de datos
    resting_blood_pressure = st.text_input("Valor de resting_blood_pressure:")
    serum_cholesterol_mg_per_dl = st.text_input("Valor de serum_cholesterol_mg_per_dl:")
    max_heart_rate_achieved = st.text_input("Valor de max_heart_rate_achieved:")
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):         
        x_in =[np.float_(resting_blood_pressure.title()),
                    np.float_(serum_cholesterol_mg_per_dl.title()),
                    np.float_(max_heart_rate_achieved.title())]
        predictS = model_prediction(x_in, model)
        st.success('La predicción de dolencia cardiaca es: {}'.format(predictS[0]).upper())

if __name__ == '__main__':
    main()
