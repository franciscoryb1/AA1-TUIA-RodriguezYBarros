# Librerias
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

# Streamlit app

class Split_data:
    def convert_date(df):
        df["Date"] = pd.to_datetime(df["Date"])
    
    def split_data(self, df):
        self.fecha_limite = "2016-01-01"
        self.df_train = df[df["Date"] < self.fecha_limite]
        self.df_test = df[df["Date"] >= self.fecha_limite]
        return self



file_path = "weatherAUS.csv"
df = pd.read_csv(file_path, sep=",", engine="python")

