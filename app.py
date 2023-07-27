# Importando as bibliotecas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import os

# Pegando o caminho base e caminho do arquivo
current_dir = os.path.dirname(os.path.abspath(__file__))
path_file = os.path.join(current_dir, r"diabetes.csv")

# Lendo a base de dados
dados = pd.read_csv(path_file)

# dropando dados nulos se existentes
dados.dropna(inplace=True)

# Dividindo os dados em entrada e saída
X = dados.drop("Outcome", axis=1)
y = dados["Outcome"]

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando o SMOTE para fazer oversampling dos dados de treino com sampling_strategy específico
sm = SMOTE(random_state=42, sampling_strategy={0: 1000, 1: 1000})
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Treinando um modelo de regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# Avaliando o modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Criando a interface do aplicativo web
st.title("Prevendo Diabetes")

# Observações do modelo informando sobre o uso para mulheres
st.header(f"""⚠️⚠️⚠️Atenção⚠️⚠️⚠️
        {os.linesep}Este app só pode ser usado por mulheres, 
        {os.linesep}haja vista que a base de dados que temos 
        {os.linesep}são de registros de mulheres com ou não diabetes""")

# Permitindo a inserção do nome do usuário pelo próprio
nome = st.text_input("Digite seu nome:")

# Escrevendo o nome do usuário que foi dado na entrada
st.write(f"Olá {nome}, seja bem-vindo(a) ao aplicativo de previsão de Diabetes!")

# Criando um gráfico de barras com as métricas do modelo
st.write("O modelo utilizado para prever diabetes é uma regressão logística. As métricas de avaliação do modelo são:")
fig, ax = plt.subplots()
ax.bar(["Acurácia", "Precisão", "Recall", "F1-score"], [acc, prec, rec, f1])
ax.set_ylim(0, 1)
st.pyplot(fig)

# Criando um formulário para receber os dados de entrada do usuário
st.write("Para fazer a previsão de diabetes, por favor preencha os seguintes dados:")
with st.form(key="form"):
    # Campos para receber os dados de entrada do usuário
    gravidez = st.number_input("Número de vezes grávida:", min_value=0)
    glicose = st.number_input("Concentração de glicose no plasma:", min_value=0)
    pressao = st.number_input("Pressão arterial diastólica (mm Hg):", min_value=0)
    pele = st.number_input("Espessura da dobra da pele do tríceps (mm):", min_value=0)
    insulina = st.number_input("Insulina sérica de 2 horas (mu U/ml):", min_value=0)
    imc = st.number_input("Índice de massa corporal (kg/m^2):", min_value=0.0)
    pedigree = st.number_input("Função pedigree de diabetes:", min_value=0.0)
    idade = st.number_input("Idade (anos):", min_value=0)

    # Criando um botão para enviar os dados
    submit_button = st.form_submit_button(label="Fazer previsão")

# Fazendo a previsão com os dados de entrada do usuário
if submit_button:
    # Criando um array com os dados de entrada do usuário
    dados = [gravidez, glicose, pressao, pele, insulina, imc, pedigree, idade]
    
    # Fazendo a previsão usando o modelo treinado
    resultado = model.predict([dados])
    
    # Mostrando o resultado da previsão
    if resultado == 0:
        st.write("Parabéns, você não está propensa a ter diabetes!")
    else:
        st.write("Atenção, você está propensa a ter diabetes! Procure um médico para orientações.")
st.write("Este é um modelo de teste, suas respostas são baseadas em cálculos matemáticos e não substituiem a opinão médica ou de alguém de fato qualificado.")
