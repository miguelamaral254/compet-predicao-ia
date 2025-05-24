import pickle
import pandas as pd

def fazer_predicao(age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal):
    # Carregar o modelo e o scaler
    with open('modelo_bolado.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Criar DataFrame com os dados de entrada
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    # Normalizar os dados de entrada
    input_scaled = scaler.transform(input_data)

    # Realizar a previsão
    prediction = model.predict(input_scaled)

    # Retornar o resultado
    return "Presença de doença cardíaca detectada." if prediction[0] == 1 else "Nenhuma presença de doença cardíaca detectada."
