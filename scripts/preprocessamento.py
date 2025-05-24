# preprocessamento.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

# Função para carregar os dados
def carregar_dados():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(url, names=column_names)
    return df

# Função para pré-processar os dados
def preprocessar_dados(df):
    # Substituir '?' por NaN e converter colunas para numérico
    df.replace('?', np.nan, inplace=True)
    df[['ca', 'thal']] = df[['ca', 'thal']].apply(pd.to_numeric)

    # Remover linhas com valores ausentes
    df.dropna(inplace=True)

    # Converter a variável alvo em binária (1 = doença cardíaca, 0 = não)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Separar features (X) e alvo (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

def salvar_grafico_idade(df):
    # Garantir que a pasta 'resultados' existe
    if not os.path.exists('resultados'):
        os.makedirs('resultados')

    print("Gerando gráfico...") 

    # Gerar o gráfico da distribuição da idade
    sns.histplot(df['age'], kde=True)
    plt.title('Distribuição da Idade')

    # Criar um nome único para o arquivo, usando a data e hora
    nome_arquivo = f"distribuicao_idade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    caminho_completo = f'resultados/{nome_arquivo}'
    
    # Salva o gráfico na pasta 'resultados'
    plt.savefig(caminho_completo)  
    print(f"Gráfico salvo em: {caminho_completo}") 

    plt.close()  

    # Salvar a análise do gráfico de idade em JSON
    analise = {
        "grafico": nome_arquivo,
        "descricao": "Distribuicao da Idade dos Pacientes"
    }

    with open('resultados/analise_distribuicao_idade.json', 'w') as json_file:
        json.dump(analise, json_file)
