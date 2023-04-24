"""
Michael Andres Casadiegos Berrio

Linear regression with the LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver method is a numerical optimization method used to find the minimum of an objective function. 
It is a gradient descent algorithm that uses an approximation of the Hessian matrix to minimize the function.
The data in the CSV file contains medical information and costs billed by health insurance companies. Its columns are: age, sex, BMI, children, smoker, region, insurance charges.

You can download the used data set here: https://www.kaggle.com/datasets/mirichoi0218/insurance?ref=hackernoon.com

"""


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from prettytable import PrettyTable

# carga de datos desde el archivo CSV
data = pd.read_csv(r'C:\Users\maico\Downloads\insurance.csv')


# separamos las características (X) y las etiquetas (y)
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].values
y = data['charges'].values

# codificación de las columnas 'sex', 'smoker' y 'region'
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 4] = le.fit_transform(X[:, 4])
X[:, 5] = le.fit_transform(X[:, 5])

# conversión a valores numéricos
X = X.astype(float)

# normalización de características
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# función de pérdida para el modelo de regresión lineal
def linear_loss(theta, X, y):
    y_pred = np.dot(X, theta)
    return np.mean((y_pred - y) ** 2)

# inicialización de los parámetros del modelo
theta_init = np.zeros(X.shape[1])

# ajuste del modelo mediante el método L-BFGS
result = minimize(linear_loss, theta_init, args=(X, y), method='L-BFGS-B')

# impresión de los parámetros ajustados del modelo
table = PrettyTable()
table.field_names = ["Características", "Parámetro ajustado"]
for i, col_name in enumerate(data.columns[:-1]):
    table.add_row([col_name, round(result.x[i], 4)])
print(table)
