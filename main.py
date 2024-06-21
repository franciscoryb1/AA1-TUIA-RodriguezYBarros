import streamlit as st
import joblib
import pandas as pd
from datetime import date
from app import TransformData20

# Cargar el modelo entrenado
st.title('Predicción de Lluvia')
pipelineRainTomorrow = 'rain_tomorrow.joblib'
pipelineRainFallTomorrow = 'rain_fall_tomorrow.joblib'

try:
    modelRainTomorrow = joblib.load(pipelineRainTomorrow)
    modelRainFallTomorrow = joblib.load(pipelineRainFallTomorrow)
    # st.sidebar.success("Pipeline cargado exitosamente.")
except FileNotFoundError:
    st.sidebar.error(f'Ocurrió un error al cargar el modelo.')
except Exception as ex:
    st.sidebar.error(f'Ocurrió un error al cargar el modelo.')


# Sidebar para información adicional o ajustes
st.sidebar.header('Ajustes')

# Obtener la fecha actual
fecha = date.today()
# Sliders para los datos de entrada
st.sidebar.subheader('Datos Meteorológicos')
minTemp = st.sidebar.slider('Temperatura mínima (°C)', -10.0, 40.0, 20.0)
maxTemp = st.sidebar.slider('Temperatura máxima (°C)', -10.0, 40.0, 20.0)
temp9am = st.sidebar.slider('Temperatura 9am (°C)', -10.0, 40.0, 20.0)
temp3pm = st.sidebar.slider('Temperatura 3pm (°C)', -10.0, 40.0, 20.0)
rainfall = st.sidebar.slider('Lluvia (mm)', 0.0, 150.0, 20.0)
evaporation = st.sidebar.slider('Evaporación (mm)', 0.0, 100.0, 50.0)
sunshine = st.sidebar.slider('Horas de sol', 0.0, 20.0, 10.0)
windGustSpeed = st.sidebar.slider('Velocidad de ráfaga de viento (km/h)', 0.0, 130.0, 50.0)
windSpeed9am = st.sidebar.slider('Velocidad de viento 9am (km/h)', 0.0, 130.0, 50.0)
windSpeed3pm = st.sidebar.slider('Velocidad de viento 3pm (km/h)', 0.0, 130.0, 50.0)
humidity9am = st.sidebar.slider('Humedad 9am (%)', 0.0, 100.0, 50.0)
humidity3pm = st.sidebar.slider('Humedad 3pm (%)', 0.0, 100.0, 50.0)
pressure9am = st.sidebar.slider('Presión 9am (hPa)', 0.0, 100.0, 50.0)
pressure3pm = st.sidebar.slider('Presión 3pm (hPa)', 0.0, 100.0, 50.0)
cloud9am = st.sidebar.slider('Nubosidad 9am (%)', 0.0, 100.0, 50.0)
cloud3pm = st.sidebar.slider('Nubosidad 3pm (%)', 0.0, 100.0, 50.0)

# Crear el DataFrame con los datos ingresados por el usuario
data = {
    'Date': [fecha],
    'MinTemp': [minTemp],
    'MaxTemp': [maxTemp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustSpeed': [windGustSpeed],
    'WindSpeed9am': [windSpeed9am],
    'WindSpeed3pm': [windSpeed3pm],
    'Humidity9am': [humidity9am],
    'Humidity3pm': [humidity3pm],
    'Pressure9am': [pressure9am],
    'Pressure3pm': [pressure3pm],
    'Cloud9am': [cloud9am],
    'Cloud3pm': [cloud3pm],
    'Temp9am': [temp9am],
    'Temp3pm': [temp3pm],
    'Bimestre': ['Bimestre 1'],  # Opción fija por ahora
}

# Mostrar los datos ingresados por el usuario
st.subheader('Datos de Entrada')
fila = pd.DataFrame(data)
st.write(fila)

# RainFallTomorrow
prediccion_rain_tomorrow = modelRainTomorrow.predict(fila)
prediccion_rain_fall_tomorrow = modelRainFallTomorrow.predict(fila)

st.subheader('Predicción RainFallTomorrow')
st.write(prediccion_rain_fall_tomorrow)

# RainTomorrow
st.subheader('Predicción RainTomorrow')
if prediccion_rain_tomorrow[0] == 1:
    st.write("¡Sí! Probablemente lloverá mañana.")
    st.image("assets/rainy_icon.png.png", width=100)
else:
    st.write("No, es probable que no llueva mañana.")
    st.image("assets/sunny_icon.png", width=100)
