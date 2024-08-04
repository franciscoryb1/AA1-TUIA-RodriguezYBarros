import streamlit as st
import pandas as pd
import joblib
from datetime import date
from app import CustomScaler, DummiesTransformer, NeuralNetworkTensorFlowRl, NeuralNetworkTensorFlowRegressor

# Título de la aplicación
st.title('Predicción de Lluvia')

# Ruta del modelo entrenado
pipelineRainTomorrow = './pipelines/pipelineRL.joblib'
pipelineRainFallTomorrow = './pipelines/pipelineRegressor.joblib'
modelRainTomorrow = joblib.load(pipelineRainTomorrow)
modelRainFallTomorrow = joblib.load(pipelineRainFallTomorrow)

# Cargar el modelo entrenado
# try:
#     modelRainTomorrow = joblib.load(pipelineRainTomorrow)
#     st.sidebar.success("Pipeline cargado exitosamente.")
# except FileNotFoundError:
#     st.sidebar.error(f'Ocurrió un error al cargar el modelo.')
#     st.stop()  # Detener la ejecución si el modelo no se carga
# except Exception as ex:
#     st.sidebar.error(f'Ocurrió un error al cargar el modelo: {ex}')
#     st.stop()

# Sidebar para ajustes
st.sidebar.header('Ajustes')

# Obtener la fecha actual
fecha = date.today()

# Sliders para los datos de entrada
st.sidebar.subheader('Datos Meteorológicos')

minTemp = st.sidebar.slider('Temperatura mínima (°C)', -10.0, 40.0, 20.0)
maxTemp = st.sidebar.slider('Temperatura máxima (°C)', -10.0, 40.0, 20.0)
rainfall = st.sidebar.slider('Lluvia (mm)', 0.0, 150.0, 20.0)
evaporation = st.sidebar.slider('Evaporación (mm)', 0.0, 100.0, 50.0)
sunshine = st.sidebar.slider('Horas de sol', 0.0, 20.0, 10.0)
WindGustDir = st.sidebar.selectbox(
    'WindGustDir',
    options=["N", "NNW", "NNE", "S", "SSW", "SSE", "E", "ENE", "ESE", "SE", "NE", "W", "WNW", "WSW", "SW", "NW"])
windGustSpeed = st.sidebar.slider('Velocidad de ráfaga de viento (km/h)', 0.0, 130.0, 50.0)
WindDir9am = st.sidebar.selectbox(
    'WindDir9am',
    options=["N", "NNW", "NNE", "S", "SSW", "SSE", "E", "ENE", "ESE", "SE", "NE", "W", "WNW", "WSW", "SW", "NW"])
WindDir3pm = st.sidebar.selectbox(
    'WindDir3pm',
    options=["N", "NNW", "NNE", "S", "SSW", "SSE", "E", "ENE", "ESE", "SE", "NE", "W", "WNW", "WSW", "SW", "NW"])
windSpeed9am = st.sidebar.slider('Velocidad de viento 9am (km/h)', 0.0, 130.0, 50.0)
windSpeed3pm = st.sidebar.slider('Velocidad de viento 3pm (km/h)', 0.0, 130.0, 50.0)
humidity9am = st.sidebar.slider('Humedad 9am (%)', 0.0, 100.0, 50.0)
humidity3pm = st.sidebar.slider('Humedad 3pm (%)', 0.0, 100.0, 50.0)
pressure9am = st.sidebar.slider('Presión 9am (hPa)', 950.0, 1050.0, 1010.0)
pressure3pm = st.sidebar.slider('Presión 3pm (hPa)', 950.0, 1050.0, 1010.0)
cloud9am = st.sidebar.slider('Nubosidad 9am (%)', 0.0, 100.0, 50.0)
cloud3pm = st.sidebar.slider('Nubosidad 3pm (%)', 0.0, 100.0, 50.0)
temp9am = st.sidebar.slider('Temperatura 9am (°C)', -10.0, 40.0, 20.0)
temp3pm = st.sidebar.slider('Temperatura 3pm (°C)', -10.0, 40.0, 20.0)
RainToday = st.sidebar.selectbox(
    'RainToday',
    options=["Si", "No"]
)

# Crear el DataFrame con los datos ingresados por el usuario
data = {
    'MinTemp': [minTemp],
    'MaxTemp': [maxTemp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustDir': [WindGustDir],
    'WindGustSpeed': [windGustSpeed],
    'WindDir9am': [WindDir9am],
    'WindDir3pm': [WindDir3pm],
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
    'RainToday': [RainToday]
}

# Mostrar los datos ingresados por el usuario
st.subheader('Datos de Entrada')
fila = pd.DataFrame(data)
st.write(fila)

# Hacer la predicción
prediccion_rain_tomorrow = modelRainTomorrow.predict(fila)
prediccion_rain_fall_tomorrow = modelRainFallTomorrow.predict(fila)

# Mostrar la predicción
st.subheader('Predicción RainTomorrow')
st.write(prediccion_rain_fall_tomorrow)
st.write(prediccion_rain_tomorrow)
if prediccion_rain_tomorrow[0] == 1:
    st.write("¡Sí! Probablemente lloverá mañana.")
    # st.image("assets/rainy_icon.png", width=100)
else:
    st.write("No, es probable que no llueva mañana.")
    # st.image("assets/sunny_icon.png", width=100)
