import streamlit as st
import joblib
import numpy as np
import pandas as pd
from app import TransformData20

st.title('Rain predictions')
joblib_file = 'pipeline.joblib'
try:
    pipeline_entrenado = joblib.load(joblib_file)
    print("Pipeline cargado exitosamente.")
except FileNotFoundError:
    print(f'El archivo {joblib_file} no existe.')
except Exception as ex:
    print(f'Ocurrio un error al cargar el pipeline: {ex}')

sepal_lenght = st.slider('Sepal lenght', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal width', 4.0, 8.0, 5.0)
petal_lenght = st.slider('petal lenght', 4.0, 8.0, 5.0)
petal_width = st.slider('petal width', 4.0, 8.0, 5.0)
