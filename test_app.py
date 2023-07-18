import joblib
import pandas as pd
import pytest
import requests
from fichier_streamlit import get_prediction



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
    assert 0 <= response_json["proba"] <= 1  # Vérifier la plage de la probabilité
    #assert isinstance(response_json["prediction"], int)  # Vérifier le type de la prédiction

    
    


# Test si la page affiche les informations du client sélectionné
def test_affichage_infos_client():
    id_client = 408267
    data_client = df.loc[df["SK_ID_CURR"] == id_client]
    assert data_client is not None

    
    
    
# Test si la prédiction et la probabilité affichées sont valides

def test_prediction_proba():
    id_client = 408267
    data_client = df.loc[df["SK_ID_CURR"] == id_client]
    X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).values
    prediction = model.predict(X)
    proba = model.predict_proba(X)[:, 1][0]
    assert (prediction == 0 or prediction == 1) and proba >= 0 and proba <= 1