# Prevendo Diabetes

Este projeto é um aplicativo web que usa Machine Learning para prever se uma pessoa tem diabetes ou não, baseado em alguns dados de entrada. O aplicativo usa Streamlit para criar a interface gráfica, pandas para manipular os dados, scikit-learn para treinar e avaliar um modelo de regressão logística, e matplotlib para gerar gráficos. A base de dados usada é a [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), que contém registros médicos de mulheres indígenas dos EUA.

## Como clonar o projeto

- Abra o terminal e navegue até a pasta onde você quer salvar o projeto.
- Digite o comando `git clone https://github.com/Gustavo-H-Martins/prevendo-diabetes.git` para clonar o repositório do GitHub.
- Digite o comando `cd prevendo-diabetes` para entrar na pasta do projeto.

## Como inicializar o aplicativo
- Digite o comando `python.exe -m venv .venv` para iniciar o ambiente virtual (se não houver instalado antes `pip install virtualenv`)
- Digite o comando `.\.venv\Scripts\activate` para iniciar o ambiente virutal
- Digite o comando `pip install -r requirements.txt` para instalar as bibliotecas necessárias.
- Digite o comando `streamlit run app.py` para iniciar o aplicativo web.
- Abra o navegador e digite o endereço `http://localhost:8501` para acessar o aplicativo.

## Como testar o aplicativo

- Na página inicial do aplicativo, digite seu nome no campo "Digite seu nome:" e clique em Enter.
- Preencha os dados solicitados no formulário, como número de vezes grávida, concentração de glicose no plasma, pressão arterial diastólica, etc.
- Clique no botão "Fazer previsão" para enviar os dados e ver o resultado da previsão.
- Veja se o resultado da previsão está de acordo com o esperado, e se a interface do aplicativo está funcionando corretamente.

## Bibliotecas usadas

- [Streamlit](https://docs.streamlit.io/library/get-started): uma biblioteca Python que permite criar aplicativos web interativos para Machine Learning. Você pode ver mais informações sobre o Streamlit neste [link](https://docs.streamlit.io/library/get-started).
- [Pandas](https://pandas.pydata.org/): uma biblioteca Python que oferece estruturas e ferramentas de alta performance para trabalhar com dados tabulares. Você pode ver mais informações sobre o Pandas neste [link](https://pandas.pydata.org/).
- [Matplotlib](https://matplotlib.org/): uma biblioteca Python que permite criar gráficos e visualizações de dados. Você pode ver mais informações sobre o Matplotlib neste [link](https://matplotlib.org/).
- [Scikit-learn](https://scikit-learn.org/stable/getting_started.html): uma biblioteca Python que oferece ferramentas simples e eficientes para Machine Learning. Você pode ver mais informações sobre o Scikit-learn neste [link](https://scikit-learn.org/stable/getting_started.html).
- [Imbalanced-learn](https://imbalanced-learn.org/stable/install.html): uma biblioteca Python que oferece técnicas para lidar com dados desbalanceados. Você pode ver mais informações sobre a Imbalanced-learn neste [link](https://imbalanced-learn.org/stable/install.html).

# Lincença de uso
- [Lincença](.\LICENCE.md)