# Librerias
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib

# Carga de dataset y limpieza inicial
file_path = "weatherAUS.csv"
df0 = pd.read_csv(file_path, sep=",", engine="python")
df = df0.copy()
ciudades = [
    " Adelaide",
    "Canberra",
    "Cobar",
    "Dartmoor",
    "Melbourne",
    "MelbourneAirport",
    "MountGambier",
    "Sydney",
    "SydneyAirport",
]  
# Filtrar por ciudades
df = df[df["Location"].isin(ciudades)]
df = df.drop("Location", axis=1)
# X
X = df.drop(['RainTomorrow', 'RainfallTomorrow', 'Unnamed: 0'], axis=1)
# y
y = df[["RainfallTomorrow"]]
# Rellenar valores faltantes de RainFallTomorrow
mediana = y['RainfallTomorrow'].median()
# Rellenar NaN con la mediana
y = y['RainfallTomorrow'].fillna(mediana)
y.isna().sum()

class TransformData20(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        return self  # No se necesita hacer ningún ajuste en el fit para este caso
    
    def transform(self, X, y=None):        
        # Eliminar columna "Unnamed: 0"
        if 'Unnamed: 0' in X.columns:
            X = X.drop("Unnamed: 0", axis=1)
        
        # Convertir la fecha a datetime
        X['Date'] = pd.to_datetime(X['Date'])
        
        # Determinar bimestre
        X['Bimestre'] = X['Date'].apply(self.determinar_bimestre)
        
        # Rellenar valores faltantes de Rainfall
        mediana_por_dia = X.groupby(X["Date"].dt.date)["Rainfall"].median()
        X["Rainfall"] = X.apply(
            lambda row: mediana_por_dia[row["Date"].date()] if pd.isnull(row["Rainfall"]) else row["Rainfall"],
            axis=1,
        )
        
        # Rellenar valores faltantes de Evaporation por bimestre
        medianas_evaporation = X.groupby("Bimestre")["Evaporation"].median()
        for bimestre, median in medianas_evaporation.items():
            X.loc[(X["Bimestre"] == bimestre) & (X["Evaporation"].isnull()), "Evaporation"] = median
        
        # Rellenar valores faltantes de Sunshine por día
        X['Sunshine'] = X.groupby(X['Date'].dt.day)["Sunshine"].transform(lambda x: x.fillna(x.mean()))


        # Rellenar valores faltantes de WindDir por día
        X["WindGustDir"] = X.groupby(X["Date"].dt.day)[
            "WindGustDir"
        ].transform(lambda x: x.fillna(x.mode().iloc[0]))
        X["WindDir9am"] = X.groupby(X["Date"].dt.day)[
            "WindDir9am"
        ].transform(lambda x: x.fillna(x.mode().iloc[0]))
        X["WindDir3pm"] = X.groupby(X["Date"].dt.day)[
            "WindDir3pm"
        ].transform(lambda x: x.fillna(x.mode().iloc[0]))
        
        # Rellenar valores faltantes de WindSpeed, Humidity, Cloud, Pressure, Temp por día
        columns_to_fillna = ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Cloud9am',
                             'Cloud3pm', 'Pressure9am', 'Pressure3pm', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']
        
        for column in columns_to_fillna:
            if column in X.columns:
                X[column] = X.groupby(X['Date'].dt.day)[column].transform(lambda x: x.fillna(x.median()))
        
        # Rellenar valores faltantes de RainToday con la moda y pasarlo a 1 y 0
        moda_RainToday = X.groupby("Date")["RainToday"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        X["RainToday"] = X["RainToday"].fillna(moda_RainToday)
        X["RainToday"] = X["RainToday"].map({"Yes": 1, "No": 0})

        # Agrupar direcciones de viento
        X['WindGustDir_Agrupado'] = X['WindGustDir'].apply(self.agrupar_direcciones)
        X['WindDir9am_Agrupado'] = X['WindDir9am'].apply(self.agrupar_direcciones)
        X['WindDir3pm_Agrupado'] = X['WindDir3pm'].apply(self.agrupar_direcciones)
        X = X.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
        
        # Crear variables dummies para direcciones agrupadas
        X = pd.get_dummies(X, columns=['WindGustDir_Agrupado', 'WindDir9am_Agrupado', 'WindDir3pm_Agrupado'],
                            drop_first=True)
        
        # Calcular diferencia de temperatura máxima y mínima
        X['Dif_Temp_Max_Min'] = X['MaxTemp'] - X['MinTemp']
        X = X.drop(['MaxTemp', 'MinTemp'], axis=1)
        
        # Calcular diferencia de temperaturas 9am y 3pm
        X['Temp_Difference'] = X['Temp3pm'] - X['Temp9am']
        X = X.drop(['Temp3pm', 'Temp9am'], axis=1)
        
        # Eliminar columnas innecesarias
        X = X.drop(['Date', 'Bimestre'], axis=1)

        # df_train
        scaler = StandardScaler()
        X_Scale = scaler.fit_transform(X)
        X_Scale = pd.DataFrame(X_Scale, columns=X.columns)
        
        return X_Scale
    
    def determinar_bimestre(self, fecha):
        mes = fecha.month
        if 1 <= mes <= 2:
            return "Bimestre 1"
        elif 3 <= mes <= 4:
            return "Bimestre 2"
        elif 5 <= mes <= 6:
            return "Bimestre 3"
        elif 7 <= mes <= 8:
            return "Bimestre 4"
        elif 9 <= mes <= 10:
            return "Bimestre 5"
        else:
            return "Bimestre 6"
    
    def agrupar_direcciones(self, direccion):
        grupos_principales = {
            "N": ["N", "NNW", "NNE"],
            "S": ["S", "SSW", "SSE"],
            "E": ["E", "ENE", "ESE", "SE", "NE"],
            "W": ["W", "WNW", "WSW", "SW", "NW"],
        }

        for grupo, direcciones in grupos_principales.items():
            if direccion in direcciones:
                return grupo

        return "Otro"

# Pipeline
pipeline = Pipeline([
    ('transform_data', TransformData20()),
    ('regression', LinearRegression())
])

# Entrenar el pipeline
pipeline.fit(X, y)

# Fila para PREDICT
pd.set_option('future.no_silent_downcasting', True)
# Filtrar el DataFrame
fila = df.iloc[10]
# Convertir la fila en un nuevo DataFrame
df_fila = pd.DataFrame(fila).transpose()
df_fila = df_fila.drop(['RainTomorrow', 'RainfallTomorrow', 'Unnamed: 0'], axis=1)
transformer = TransformData20()
transformer.transform(df_fila)

# Predecir con el pipeline
predictions = pipeline.predict(df_fila)
print(predictions)

# joblib.dump(pipeline, 'pipeline.joblib')