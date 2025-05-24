# preprocessamento.py (atualizado)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

def carregar_dados():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(url, names=column_names)
    return df

def preprocessar_dados(df):
    # Substituir '?' por NaN e converter colunas para numérico
    df.replace('?', np.nan, inplace=True)
    df[['ca', 'thal']] = df[['ca', 'thal']].apply(pd.to_numeric)

    # Remover linhas com valores ausentes
    df.dropna(inplace=True)

    # Converter a variável alvo em binária
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Separar features e alvo
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y, df

def gerar_visualizacoes(df):
    """Gera e salva múltiplas visualizações dos dados"""
    if not os.path.exists('resultados/visualizacoes'):
        os.makedirs('resultados/visualizacoes')
    
    analises = {}
    
    # 1. Distribuição da idade por doença cardíaca
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y='age', data=df)
    plt.title('Distribuição da Idade por Presença de Doença Cardíaca')
    plt.xlabel('Doença Cardíaca (0=Não, 1=Sim)')
    plt.ylabel('Idade')
    idade_path = 'resultados/visualizacoes/idade_target.png'
    plt.savefig(idade_path)
    plt.close()
    analises['idade_target'] = idade_path
    
    # 2. Correlação entre variáveis
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlação')
    corr_path = 'resultados/visualizacoes/matriz_correlacao.png'
    plt.savefig(corr_path)
    plt.close()
    analises['correlacao'] = corr_path
    
    # 3. Distribuição das principais variáveis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(df['trestbps'], kde=True, ax=axes[0, 0])
    sns.histplot(df['chol'], kde=True, ax=axes[0, 1])
    sns.histplot(df['thalach'], kde=True, ax=axes[1, 0])
    sns.countplot(x='sex', hue='target', data=df, ax=axes[1, 1])
    plt.tight_layout()
    dist_path = 'resultados/visualizacoes/distribuicoes.png'
    plt.savefig(dist_path)
    plt.close()
    analises['distribuicoes'] = dist_path
    
    # Salvar metadados das análises
    with open('resultados/analise_visualizacoes.json', 'w') as f:
        json.dump(analises, f, indent=4)