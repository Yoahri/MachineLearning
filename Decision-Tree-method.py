import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Carregar os dados do arquivo CSV
data = pd.read_csv('reg02.csv')

# Separar os recursos (X) e o alvo (y)
X = data.drop('target', axis=1)  # Substitua 'target_column_name' pelo nome da coluna alvo
y = data['target']

# Definir o número de folds
n_splits = 5

# Inicializar o modelo de árvore de decisão
model = DecisionTreeRegressor(random_state=42)

# Inicializar o objeto KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listas para armazenar os MAEs de treino e validação
train_maes = []
validation_maes = []

# Loop sobre os conjuntos de treino/teste
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões nos conjuntos de treino e validação
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    # Calcular o MAE para os conjuntos de treino e validação

    train_mae = mean_absolute_error(y_train, train_preds)
    validation_mae = mean_absolute_error(y_val, val_preds)

    # Armazenar os resultados
    train_maes.append(train_mae)
    validation_maes.append(validation_mae)

# Calcular a média dos MAEs de treino e validação
mean_train_mae = sum(train_maes) / len(train_maes)
mean_validation_mae = sum(validation_maes) / len(validation_maes)

print(f'MAE médio para a base de treino: {mean_train_mae}')
print(f'MAE médio para a base de validação: {mean_validation_mae}')
print(data)