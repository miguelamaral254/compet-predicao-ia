# modelo.py
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import os
import unicodedata

# Função para remover acentos e caracteres especiais
def remover_acentos(texto):
    nfkd_form = unicodedata.normalize('NFKD', texto)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def treinar_modelo(X, y):
    log_accuracies = []
    tree_accuracies = []
    
    # Para armazenar o relatório de classificação para cada modelo
    log_classification_reports = []
    tree_classification_reports = []

    for seed in range(30):
        # Divisão dos dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Normalização dos dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Treinamento dos modelos
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train_scaled, y_train)
        log_preds = log_model.predict(X_test_scaled)
        log_acc = accuracy_score(y_test, log_preds)
        log_accuracies.append(log_acc)
        log_classification_reports.append(classification_report(y_test, log_preds, output_dict=True))

        tree_model = DecisionTreeClassifier()
        tree_model.fit(X_train_scaled, y_train)
        tree_preds = tree_model.predict(X_test_scaled)
        tree_acc = accuracy_score(y_test, tree_preds)
        tree_accuracies.append(tree_acc)
        tree_classification_reports.append(classification_report(y_test, tree_preds, output_dict=True))

    # Cálculo de média e desvio padrão das acurácias
    log_mean = np.mean(log_accuracies)
    log_std = np.std(log_accuracies)
    tree_mean = np.mean(tree_accuracies)
    tree_std = np.std(tree_accuracies)

    # Seleção do melhor modelo com base na média de acurácia
    if log_mean > tree_mean:
        modelo_bolado = LogisticRegression(max_iter=1000)
        model_name = "Regressao Logistica"
        classification_report_data = log_classification_reports
    else:
        modelo_bolado = DecisionTreeClassifier()
        model_name = "Arvore de Decisao"
        classification_report_data = tree_classification_reports

    # Treinamento do melhor modelo com todos os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelo_bolado.fit(X_scaled, y)

    # Salvamento do modelo e do scaler
    with open('modelo_bolado.pkl', 'wb') as f:
        pickle.dump(modelo_bolado, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Garantir que a pasta 'resultados' existe
    if not os.path.exists('resultados'):
        os.makedirs('resultados')

    # Salvar a análise das acurácias e do relatório de classificação em JSON
    analise = {
        "modelo": remover_acentos(model_name),
        "acuracia_media_regressao_logistica": log_mean,
        "acuracia_desvio_regressao_logistica": log_std,
        "acuracia_media_arvore_decisao": tree_mean,
        "acuracia_desvio_arvore_decisao": tree_std,
        "relatorio_classificacao": classification_report_data
    }

    # Definir o caminho completo do arquivo JSON
    caminho_json = 'resultados/analise_modelo_completo.json'

    # Salvando as métricas e o relatório de classificação no arquivo JSON
    with open(caminho_json, 'w', encoding='utf-8') as json_file:
        json.dump(analise, json_file, ensure_ascii=False, indent=4)

    # Exibir as métricas no console em formato percentual
    print(f"\nMétricas de Acurácia dos Modelos (em %):\n")
    print(f"Regressão Logística - Média: {log_mean * 100:.2f}%, Desvio Padrão: {log_std * 100:.2f}%")
    print(f"Árvore de Decisão - Média: {tree_mean * 100:.2f}%, Desvio Padrão: {tree_std * 100:.2f}%")

    # Exibir o relatório de classificação para ambos os modelos
    print(f"\nRelatório de Classificação - {model_name} (em %):\n")
    for class_label in ['0', '1']:
        print(f"{class_label} - Precision: {classification_report_data[0][class_label]['precision'] * 100:.2f}%, "
              f"Recall: {classification_report_data[0][class_label]['recall'] * 100:.2f}%, "
              f"F1-Score: {classification_report_data[0][class_label]['f1-score'] * 100:.2f}%")

    # Informar o caminho do arquivo JSON
    print(f"\nAnálise salva em: {caminho_json}\n")

    return model_name
