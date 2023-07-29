import shap
import pytest
import requests_mock
import pandas as pd
import numpy as np
import requests
import os

from fichier_streamlit import get_prediction, is_online_api_available

# Variable for testing
id_client = 425479

# Define the URLS
local_url = f"http://localhost:8000/predict/{id_client}"
online_url = f"https://fast-api-dashboard-final.onrender.com/predict/{id_client}"

# Use the online url if available, else the local one
url = online_url if is_online_api_available(online_url) else local_url

def test_get_prediction_response():
    response = requests.get(url)
    assert response.status_code == 200
    if response.status_code == 200:
        prediction_data = response.json()
    else:
        print(f"Request failed with status {response.status_code}")

def test_get_prediction():
    with requests_mock.Mocker() as m:
        m.get(url, json={'probaClasse0': 0.3, 'probaClasse1': 0.7})
        prediction_data = get_prediction(id_client)
        assert prediction_data['probaClasse0'] == 0.3
        assert prediction_data['probaClasse1'] == 0.7
        



