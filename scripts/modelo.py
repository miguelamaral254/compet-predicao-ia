# modelo.py (atualizado)
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
import json
import os
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns

def remover_acentos(texto):
    nfkd_form = unicodedata.normalize('NFKD', texto)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def treinar_modelo(X, y):
    # Configurações
    n_execucoes = 30
    modelos = {
        'Regressao Logistica': LogisticRegression(max_iter=1000),
        'Arvore de Decisao': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    resultados = {nome: {'acuracia': [], 'roc_auc': [], 'matrizes': []} 
                 for nome in modelos}
    
    for seed in range(n_execucoes):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for nome, modelo in modelos.items():
            # Treinamento e predição
            modelo.fit(X_train_scaled, y_train)
            preds = modelo.predict(X_test_scaled)
            probas = modelo.predict_proba(X_test_scaled)[:, 1]
            
            # Armazenar métricas
            resultados[nome]['acuracia'].append(
                accuracy_score(y_test, preds))
            resultados[nome]['roc_auc'].append(
                roc_auc_score(y_test, probas))
            resultados[nome]['matrizes'].append(
                confusion_matrix(y_test, preds))
    
    # Análise dos resultados
    analise = {}
    for nome in modelos:
        # Métricas agregadas
        analise[nome] = {
            'acuracia_media': np.mean(resultados[nome]['acuracia']),
            'acuracia_desvio': np.std(resultados[nome]['acuracia']),
            'roc_auc_medio': np.mean(resultados[nome]['roc_auc']),
            'roc_auc_desvio': np.std(resultados[nome]['roc_auc']),
            'matriz_confusao_media': np.mean(
                resultados[nome]['matrizes'], axis=0).tolist()
        }
    
    # Seleção do melhor modelo
    melhor_modelo_nome = max(modelos.keys(), 
        key=lambda x: analise[x]['acuracia_media'])
    melhor_modelo = modelos[melhor_modelo_nome]
    
    # Treinar o melhor modelo com todos os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    melhor_modelo.fit(X_scaled, y)
    
    # Salvar modelo e scaler
    with open('modelo_bolado.pkl', 'wb') as f:
        pickle.dump(melhor_modelo, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Salvar matriz de confusão como imagem
    matriz_media = np.array(analise[melhor_modelo_nome]['matriz_confusao_media'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_media, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Não Doente', 'Doente'],
                yticklabels=['Não Doente', 'Doente'])
    plt.title('Matriz de Confusão Média')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    matriz_path = 'resultados/matriz_confusao.png'
    plt.savefig(matriz_path)
    plt.close()
    analise['matriz_confusao_path'] = matriz_path
    
    # Salvar análise completa
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    
    with open('resultados/analise_completa.json', 'w', encoding='utf-8') as f:
        json.dump(analise, f, ensure_ascii=False, indent=4)
    
    # Exibir resultados
    print("\n=== RESULTADOS DOS MODELOS ===")
    for nome in modelos:
        print(f"\n{nome}:")
        print(f"Acurácia: {analise[nome]['acuracia_media']:.2%} ± {analise[nome]['acuracia_desvio']:.2%}")
        print(f"ROC AUC: {analise[nome]['roc_auc_medio']:.2%} ± {analise[nome]['roc_auc_desvio']:.2%}")
    
    print(f"\nMelhor modelo: {melhor_modelo_nome}")
    
    return melhor_modelo_nome