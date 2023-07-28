import pytest
import requests_mock
import pandas as pd
import numpy as np
import os

from fichier_streamlit import get_prediction  

def test_get_prediction():
    with requests_mock.Mocker() as m:
        m.get('https://fast-api-dashboard-final.onrender.com/predict/123', json={'probaClasse0': 0.3, 'probaClasse1': 0.7})
        prediction_data = get_prediction(123)
        assert prediction_data['probaClasse0'] == 0.3
        assert prediction_data['probaClasse1'] == 0.7
        
def test_get_prediction_response():
    response = requests.get("https://fast-api-dashboard-final.onrender.com/predict/123")
    assert response.status_code == 200


