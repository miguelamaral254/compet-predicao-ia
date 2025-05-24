# app.py
from scripts.preprocessamento import carregar_dados, preprocessar_dados, salvar_grafico_idade
from scripts.modelo import treinar_modelo
from scripts.predicoes import fazer_predicao
import gradio as gr

def main():
    # Carregar os dados
    df = carregar_dados()

    # Pré-processar os dados
    X, y = preprocessar_dados(df)

    # Gerar e salvar o gráfico da distribuição da idade
    salvar_grafico_idade(df)

    # Treinar o modelo e obter o nome do modelo
    model_name = treinar_modelo(X, y)

    # Função para previsão
    def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal):
        return fazer_predicao(age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal)

    # Criação da interface com Gradio
    interface = gr.Interface(
        fn=predict_heart_disease,
        inputs=[
            gr.Number(label="Idade"),
            gr.Radio(choices=[0, 1], label="Sexo (0 = feminino, 1 = masculino)"),
            gr.Radio(choices=[0, 1, 2, 3], label="Tipo de dor no peito (0-3)"),
            gr.Number(label="Pressão arterial em repouso"),
            gr.Number(label="Colesterol sérico"),
            gr.Radio(choices=[0, 1], label="Açúcar no sangue em jejum > 120 mg/dl"),
            gr.Radio(choices=[0, 1, 2], label="Resultados do eletrocardiograma em repouso"),
            gr.Number(label="Frequência cardíaca máxima alcançada"),
            gr.Radio(choices=[0, 1], label="Angina induzida por exercício"),
            gr.Number(label="Depressão do segmento ST induzida por exercício"),
            gr.Radio(choices=[0, 1, 2], label="Inclinação do segmento ST"),
            gr.Number(label="Número de vasos principais coloridos por fluoroscopia"),
            gr.Radio(choices=[3, 6, 7], label="Thalassemia (3 = normal, 6 = defeito fixo, 7 = defeito reversível)")
        ],
        outputs="text",
        title="Preditor de Doença Cardíaca",
        description="Insira os dados clínicos do paciente para prever a presença de doença cardíaca."
    )

    # Lançar a interface Gradio
    interface.launch()

if __name__ == "__main__":
    main()
