import numpy as np
import pandas as pd
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

# Carregar os dados do arquivo CSV
data = pd.read_csv('reg01.csv')

# Separar os recursos (X) e o alvo (y)
x = data.drop('target', axis = 1)  
y = data['target']

# Inicialize o modelo LASSO com alpha=1
lasso_model = Lasso(alpha = 1)

# Inicialize o Leave-One-Out
loo = LeaveOneOut()

# Listas para armazenar os resultados
train_rmse_results = []
validation_rmse_results = []

# Loop sobre os conjuntos de treino e teste
for train_index, test_index in loo.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Treine o modelo
    lasso_model.fit(x_train, y_train)

    # Faça previsões nos conjuntos de treino e teste
    y_train_pred = lasso_model.predict(x_train)
    y_test_pred = lasso_model.predict(x_test)

    # Calcule o RMSE para treino e teste e os armazene
    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
    train_rmse_results.append(train_rmse)

    validation_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    validation_rmse_results.append(validation_rmse)


# Calcule a média dos RMSEs
mean_train_rmse = np.mean(train_rmse_results)
mean_validation_rmse = np.mean(validation_rmse_results)

print(f'Média do RMSE para a base de treino: {mean_train_rmse}')
print(f'Média do RMSE para a base de validação: {mean_validation_rmse}')