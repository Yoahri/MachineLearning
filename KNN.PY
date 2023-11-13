import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Passo 1: Carregar os dados
data = pd.read_csv('class02.csv')

# Número de folds
k_folds = 5
fold_size = len(data) // k_folds

# Inicializar o KNN com k = 10
knn = KNeighborsClassifier(n_neighbors=10)

accuracies = []

# Passo 2: Dividir os dados e realizar a validação cruzada
for i in range(k_folds):
    # Calcular os índices para o fold de validação
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size
    valid_index = list(range(start_idx, end_idx))

    # Calcular os índices para o fold de treinamento
    train_index = [idx for idx in range(len(data)) if idx not in valid_index]

    train_data, valid_data = data.iloc[train_index], data.iloc[valid_index]
    X_train, y_train = train_data.drop('target', axis=1), train_data['target']
    X_valid, y_valid = valid_data.drop('target', axis=1), valid_data['target']

    # Passo 3: Treinar o modelo KNN e avaliá-lo
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_valid)

    # Passo 4: Calcular a acurácia para este fold
    accuracy = accuracy_score(y_valid, y_pred)
    accuracies.append(accuracy)

# Passo 5: Calcular a acurácia média
average_accuracy = sum(accuracies) / len(accuracies)

print(f'Acurácia média: {average_accuracy:.2f}')
