# predicoes.py (atualizado)
import pickle
import pandas as pd
import numpy as np

def fazer_predicao(age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal):
    try:
        # Carregar modelo e scaler
        with open('modelo_bolado.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Criar DataFrame
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        
        # Pré-processamento
        input_scaled = scaler.transform(input_data)
        
        # Predição
        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)[0][1]
        
        # Resultado detalhado
        if prediction[0] == 1:
            return (f"🔴 Risco de doença cardíaca detectado ({proba:.1%} de probabilidade). "
                    "Recomenda-se consulta médica.")
        else:
            return (f"🟢 Baixo risco de doença cardíaca ({1-proba:.1%} de probabilidade). "
                    "Mantenha hábitos saudáveis.")
    
    except Exception as e:
        return f"Erro ao processar a predição: {str(e)}"