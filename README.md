# compet-predicao-ia
Desenvolver um sistema de predição com base em técnicas de Aprendizado de Máquina, simulando uma aplicação prática de IA. O projeto será dividido em dois níveis (o aluno cumprindo o nível I já é suficiente para ser avaliado).

# Documentação Técnica do Sistema de Predição de Doenças Cardíacas

`modelo.py`

O módulo `modelo.py` implementa o núcleo da solução de machine learning, com escolhas técnicas bem fundamentadas. Para a seleção de algoritmos, optamos por três abordagens complementares:

- **Regressão Logística**: Escolhida como principal candidata devido à sua comprovada eficácia em problemas de classificação binária e à excelente interpretabilidade dos resultados, crucial para aplicações médicas.
- **Árvore de Decisão**: Utilizada como baseline, oferecendo simplicidade e transparência na interpretação.
- **Random Forest**: Método ensemble, conhecido por sua robustez e melhor desempenho em geral.

### Pré-processamento

O pré-processamento segue rigorosos padrões:
- **Remoção de valores faltantes** (representados por `?`) para garantir a qualidade dos dados.
- **Padronização** via `StandardScaler` para assegurar que modelos sensíveis à escala operem corretamente.
- **Conversão do target** para binário (0/1) para facilitar a interpretação clínica.

`Pré-processamento.py`

Este módulo é responsável pela transformação inicial dos dados. As principais abordagens implementadas são:

- **Tratamento de valores faltantes**: A estratégia adotada é a remoção completa de linhas problemáticas, considerando o tamanho adequado do dataset.
- **Visualizações**: Geramos distribuições, boxplots e matrizes de correlação, para revelar insights sobre a relação entre variáveis clínicas e a condição cardíaca. Isso segue as melhores práticas de análise exploratória.

 `predicoes.py`

A implementação das predições prioriza a robustez, com tratamento de erros completo e mensagens claras para o usuário final. A saída inclui:

- **Classificação binária** (0 ou 1).
- **Probabilidades calibradas**, oferecendo maior nuance para decisões clínicas.

A estrutura foi pensada para fácil integração com sistemas hospitalares.

## Arquivo: `app.py`

A interface Gradio foi selecionada por equilibrar simplicidade e funcionalidade. As principais características incluem:

- **Componentes de entrada**: Sliders para valores contínuos e radios para categóricos, mimetizando formulários médicos tradicionais e reduzindo a curva de aprendizado.
- **Exemplos pré-definidos**: Aceleram o teste do sistema.
- **Layout organizado em colunas**: Melhorando a experiência do usuário.

## Estratégia de Avaliação

Implementamos um rigoroso protocolo de 30 execuções independentes para garantir que nossas métricas representem desempenho realístico. As principais métricas incluem:

- **Acurácia** e **ROC AUC**, além da análise da matriz de confusão média, fornecendo insights sobre os tipos de erro do modelo.
- A avaliação multivariada considera tanto a capacidade discriminativa do modelo (ROC AUC) quanto sua consistência (desvios padrão), superando a simples acurácia.

## Armazenamento e Serialização

- **Pickle**: Usado para serialização dos modelos, balanceando eficiência e facilidade de uso.
- **JSON**: Armazenamento de metadados e resultados, garantindo legibilidade universal e facilitando auditorias.
- **PNG**: Persistência de visualizações, permitindo sua inclusão em relatórios médicos e garantindo portabilidade máxima.

## Justificativa Global

Cada decisão técnica foi pautada por três pilares:

1. **Robustez**: Através de múltiplas execuções e métricas complementares.
2. **Reprodutibilidade**: Com seeds fixos e serialização completa.
3. **Transparência**: Com visualizações detalhadas e relatórios completos.

O sistema resultante atende tanto aos requisitos técnicos de um modelo de machine learning quanto às necessidades práticas de profissionais de saúde, oferecendo previsões confiáveis em uma interface acessível. A documentação gerada automaticamente (gráficos, métricas e relatórios JSON) permite:

- **Monitoramento contínuo** do desempenho.
- **Explicação das decisões do modelo** para partes interessadas não técnicas.

