# app.py (atualizado)
from scripts.preprocessamento import carregar_dados, preprocessar_dados, gerar_visualizacoes
from scripts.modelo import treinar_modelo
from scripts.predicoes import fazer_predicao
import gradio as gr
import pandas as pd

def main():
    # Carregar e processar dados
    df = carregar_dados()
    X, y, df_processed = preprocessar_dados(df)
    
    # Gerar visualizações
    gerar_visualizacoes(df_processed)
    
    # Treinar modelo
    model_name = treinar_modelo(X, y)
    
    # Criar interface
    with gr.Blocks(title="Preditor de Doença Cardíaca") as interface:
        gr.Markdown("# Preditor de Doença Cardíaca")
        gr.Markdown(f"Modelo atual: **{model_name}**")
        
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Idade")
                sex = gr.Radio([0, 1], label="Sexo (0 = Feminino, 1 = Masculino)")
                cp = gr.Radio([0, 1, 2, 3], label="Tipo de dor no peito")
                trestbps = gr.Slider(90, 200, label="Pressão arterial em repouso")
                chol = gr.Slider(120, 600, label="Colesterol sérico")
                fbs = gr.Radio([0, 1], label="Açúcar no sangue > 120 mg/dl")
                
            with gr.Column():
                restecg = gr.Radio([0, 1, 2], label="Eletrocardiograma em repouso")
                thalach = gr.Slider(70, 220, label="Frequência cardíaca máxima")
                exang = gr.Radio([0, 1], label="Angina induzida por exercício")
                oldpeak = gr.Slider(0, 6.2, step=0.1, label="Depressão ST induzida")
                slope = gr.Radio([0, 1, 2], label="Inclinação do segmento ST")
                ca = gr.Slider(0, 3, step=1, label="Vasos principais coloridos")
                thal = gr.Radio([3, 6, 7], label="Thalassemia")
        
        submit_btn = gr.Button("Prever")
        output = gr.Textbox(label="Resultado")
        
        # Exemplos
        examples = [
            [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3],  # Não doente
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 6]   # Doente
        ]
        
        submit_btn.click(
            fn=fazer_predicao,
            inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal],
            outputs=output
        )
        
        gr.Examples(
            examples=examples,
            inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal],
            outputs=output,
            fn=fazer_predicao,
            cache_examples=True
        )
    
    interface.launch()

if __name__ == "__main__":
    main()