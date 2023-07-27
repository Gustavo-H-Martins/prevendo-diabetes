# Importando as bibliotecas
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Pegando o caminho base e caminho do arquivo
current_dir = os.path.dirname(os.path.abspath(__file__))
path_file = os.path.join(current_dir, r"diabetes.csv")

# Lendo a base de dados
dados = pd.read_csv(path_file)

# Dividindo os dados em entrada e saída
X = dados.drop("Outcome", axis=1)
y = dados["Outcome"]

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selecionando as melhores características usando o teste de qui-quadrado
selector = SelectKBest(chi2, k=4)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Treinando um modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazendo a validação cruzada com 10 folds
scores = cross_val_score(model, X_train, y_train, cv=10)
print("Acurácia média na validação cruzada:", scores.mean())

# Fazendo a previsão com o conjunto de teste
y_pred = model.predict(X_test)

# Avaliando o modelo com métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Acurácia no teste:", acc)
print("Precisão no teste:", prec)
print("Recall no teste:", rec)
print("F1-score no teste:", f1)
