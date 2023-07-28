import joblib
import pandas as pd
import pytest
import requests
from fichier_streamlit import get_prediction




df = pd.read_csv('df_dash.csv')
model = joblib.load('final_model.joblib')

url = "http://localhost:8000/predict/"  # URL de l'API FastAPI en local
#url = "https://fast-api-dashboard-final.onrender.com/
# URL de l'API FastAPI en ligne





def test_api():
    id_client = 425479  # Sélectionner un identifiant client

    response = requests.get(url + str(id_client))
    assert response.status_code == 200  # Vérifier le code de statut HTTP

    response_json = response.json()
    print(response_json)
    #assert response_json["id_client"] == id_client  # Vérifier l'identifiant client dans la réponse
    assert 0 <= response_json["probaClasse0"] or response_json["probaClasse0"]<= 1  # Vérifier la plage de la probabilité
    #assert response_json["prediction"] == 0 or response_json["prediction"] == 1
    
# def test_get_prediction():
#     # Tester la fonction de prédiction avec un exemple d'entrée
#     id_client = 425479
#     prediction = get_prediction(id_client)
#     assert isinstance(prediction, float)  # Vérifier que la prédiction est un nombre à virgule flottante
#     assert prediction >= 0.0 and prediction <= 1.0  # Vérifier que la prédiction est dans la plage attendue
    
pytest.main()




