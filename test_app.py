import joblib
import pandas as pd
import pytest
import requests




df = pd.read_csv('df_dash.csv')
model = joblib.load('final_model.joblib')

#url = "http://localhost:8000/predict/"  # URL de l'API FastAPI en local
url = "https://fast-api-dashboard-final.onrender.com/predict/"   # URL de l'API FastAPI en ligne



def test_api():
    id_client = 408267  # Sélectionner un identifiant client

    response = requests.get(url + str(id_client))
    assert response.status_code == 200  # Vérifier le code de statut HTTP

    response_json = response.json()
    print(response_json)
    #assert response_json["id_client"] == id_client  # Vérifier l'identifiant client dans la réponse
    assert 0 <= response_json["proba"] or response_json["proba"]<= 1  # Vérifier la plage de la probabilité
    #assert response_json["prediction"] == 0 or response_json["prediction"] == 1
pytest.main()




