import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    RidgeCV,
    ElasticNetCV,
    LassoCV,
    SGDRegressor,
    LogisticRegression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
     mean_squared_error, 
     r2_score, 
     mean_absolute_error,
     classification_report, 
     confusion_matrix,
     ConfusionMatrixDisplay,
     balanced_accuracy_score, 
     log_loss,
     roc_curve, 
     roc_auc_score, 
     auc,
     accuracy_score
)
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score as sklearn_f1_score
import pandas as pd
import numpy as np
import tensorflow as tf
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
import joblib

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columnas_numericas = None
        self.columnas_categoricas = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

    def fit(self, X, y=None):
        if 'Date' in X.columns:
            self.columnas_categoricas.append('Date')
        self.columnas_numericas = X.columns.difference(self.columnas_categoricas)
        X_numeric = X[self.columnas_numericas]
        self.scaler.fit(X_numeric)
        return self

    def transform(self, X, y=None):
        X_numeric = X[self.columnas_numericas]
        X_escalado_numeric = self.scaler.transform(X_numeric)
        X_esc = pd.DataFrame(X_escalado_numeric, columns=self.columnas_numericas, index=X.index)
        X_scaled = X[self.columnas_categoricas].join(X_esc)
        return X_scaled

class DummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        X = X.copy()

        # Agrupar direcciones y convertir a dummies
        X["RainToday"] = X["RainToday"].map({"Yes": 1, "No": 0}).astype(float)
        X["WindGustDir_Agrupado"] = X["WindGustDir"].apply(self.agrupar_direcciones)
        X["WindDir9am_Agrupado"] = X["WindDir9am"].apply(self.agrupar_direcciones)
        X["WindDir3pm_Agrupado"] = X["WindDir3pm"].apply(self.agrupar_direcciones)

        # Convertir a dummies
        X = self.convertir_a_dummies(X, "WindGustDir_Agrupado", "WindGustDir")
        X = self.convertir_a_dummies(X, "WindDir9am_Agrupado", "WindDir9am")
        X = self.convertir_a_dummies(X, "WindDir3pm_Agrupado", "WindDir3pm")

        # Eliminar las columnas originales
        X = X.drop(["WindGustDir", "WindDir9am", "WindDir3pm"], axis=1)

        # Guardar las columnas generadas durante el fit
        self.columns_ = X.columns
        
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Aplicar las mismas transformaciones que en fit
        X["RainToday"] = X["RainToday"].map({"Yes": 1, "No": 0}).astype(float)
        X["WindGustDir_Agrupado"] = X["WindGustDir"].apply(self.agrupar_direcciones)
        X["WindDir9am_Agrupado"] = X["WindDir9am"].apply(self.agrupar_direcciones)
        X["WindDir3pm_Agrupado"] = X["WindDir3pm"].apply(self.agrupar_direcciones)

        # Convertir a dummies
        X = self.convertir_a_dummies(X, "WindGustDir_Agrupado", "WindGustDir")
        X = self.convertir_a_dummies(X, "WindDir9am_Agrupado", "WindDir9am")
        X = self.convertir_a_dummies(X, "WindDir3pm_Agrupado", "WindDir3pm")

        # Eliminar las columnas originales
        X = X.drop(["WindGustDir", "WindDir9am", "WindDir3pm"], axis=1)
        
        # Asegurarse de que todas las columnas de fit están presentes en la transformación
        for col in self.columns_:
            if col not in X.columns:
                X[col] = 0
        
        # Reordenar las columnas como en el fit
        X = X[self.columns_]
        
        return X

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

    def convertir_a_dummies(self, X, columna_agrupada, prefijo):
        dummies = pd.get_dummies(X[columna_agrupada], dtype=int, drop_first=True)
        dummies = dummies.rename(columns={
            "N": f"{prefijo}_N", 
            "S": f"{prefijo}_S", 
            "W": f"{prefijo}_W"
        })
        X = X.drop(columna_agrupada, axis=1)
        X = pd.concat([X, dummies], axis=1)
        return X

from sklearn.utils.class_weight import compute_class_weight

class NeuralNetworkTensorFlowRl(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=32, epochs=10, learning_rate=0.001, dropout_rate=0.3, n_units_layer_0=64, n_units_layer_1=32, class_weight=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_units_layer_0 = n_units_layer_0
        self.n_units_layer_1 = n_units_layer_1
        self.class_weight = class_weight
        self.model = None

    def build_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_shape,)))
        model.add(tf.keras.layers.Dense(self.n_units_layer_0, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.n_units_layer_1, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='binary_crossentropy', 
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model


    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X)
        y = np.array(y).ravel()  # Asegurarse de que y es un array 1D

        self.model = self.build_model(X.shape[1])

        # Calculando los pesos de las clases si no están proporcionados
        if self.class_weight is None:
            classes = np.unique(y)  # Obtener las clases únicas
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            # Ajustar manualmente el peso de la clase minoritaria
            class_weights[1] = class_weights[1] * 2  # Doblar el peso de la clase 1, puedes ajustar este valor
            self.class_weight = dict(enumerate(class_weights))

        # Ajustando el modelo con class_weight
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, class_weight=self.class_weight)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.array(X)
        return self.model.predict(X)

class NeuralNetworkTensorFlowRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, batch_size=32, epochs=10, learning_rate=0.001, dropout_rate=0.3, n_units_layer_0=30):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_units_layer_0 = n_units_layer_0
        self.model = None

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(self.n_units_layer_0, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='mean_squared_error', 
                      metrics=['mse'])
        return model

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X)
        y = np.array(y).ravel()  # Asegurarse de que y es un array 1D

        self.model = self.build_model(X.shape[1])

        # Ajustando el modelo sin class_weight ya que es regresión
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = self.model.predict(X)
        return predictions.flatten()