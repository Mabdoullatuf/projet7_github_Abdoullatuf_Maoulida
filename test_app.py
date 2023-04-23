import pytest
import requests
from fichier_streamlit import get_prediction


# Test si l'API retourne une réponse valide pour un id_client donné
def test_api_response():
    id_client = 392266
    
    #https://fast-api-dashboard-final.onrender.com/docs#/default/predict_predict__id_client__get
    
    #response = requests.get(f"http://localhost:8000/predict/{id_client}")
    
    response = requests.get(f"https://fast-api-dashboard-final.onrender.com/{id_client}")
    assert response.status_code == 200
    assert response.json()["id_client"] == id_client

# Test si la fonction get_prediction retourne une probabilité valide pour un id_client donné
def test_get_prediction():
    id_client = 392266
    proba = get_prediction(id_client)
    assert proba >= 0 and proba <= 1

# # Test si la page affiche les informations du client sélectionné
# def test_affichage_infos_client():
#     id_client = 392266
#     data_client = df.loc[df["SK_ID_CURR"] == id_client]
#     assert data_client is not None

# # Test si la prédiction et la probabilité affichées sont valides
# def test_prediction_proba():
#     id_client = 392266
#     data_client = df.loc[df["SK_ID_CURR"] == id_client]
#     X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).values
#     prediction = model.predict(X)
#     proba = model.predict_proba(X)[:, 1][0]
#     assert (prediction == 0 or prediction == 1) and proba >= 0 and proba <= 1