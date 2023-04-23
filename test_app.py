import joblib
import pytest
import pandas as pd
import pytest
import requests
from fichier_streamlit import get_prediction

@pytest.fixture(scope='module')
def load_model():
    model = joblib.load('LGBM_best_model.joblib')
    return model

@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv('df_dash_10.csv')
    return data


# Test si l'API retourne une réponse valide pour un id_client donné
def test_api_response():
    id_client = 392266
    
    #https://streamlit-app-p7.onrender.com
    
    #response = requests.get(f"http://localhost:8000/predict/{id_client}")
    
    response = requests.get(f"https://streamlit-app-p7.onrender.com/{id_client}")
    assert response.status_code == 200
    assert response.json()["id_client"] == id_client

# Test si la fonction get_prediction retourne une probabilité valide pour un id_client donné
def test_get_prediction():
    id_client = 392266
    proba = get_prediction(id_client)
    assert proba >= 0 and proba <= 1

# Test si la page affiche les informations du client sélectionné
def test_affichage_infos_client():
    id_client = 392266
    data_client = load_data.loc[load_data["SK_ID_CURR"] == id_client]
    assert data_client is not None

# Test si la prédiction et la probabilité affichées sont valides
def test_prediction_proba():
    id_client = 392266
    data_client = load_data.loc[load_data["SK_ID_CURR"] == id_client]
    X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).values
    prediction = load_model.predict(X)
    proba = load_model.predict_proba(X)[:, 1][0]
    assert (prediction == 0 or prediction == 1) and proba >= 0 and proba <= 1