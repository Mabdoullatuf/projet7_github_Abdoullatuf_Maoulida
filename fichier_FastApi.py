from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import requests


# Créer une instance de l'application FastAPI
app = FastAPI()

# Charger le modèle LGBMClassifier pré-entraîné et enregistré
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger les données
df = pd.read_csv('df_dash.csv')

# Modèle de données pour l'entrée de l'API
class ClientInput(BaseModel):
    id_client: int

# Endpoint pour l'API FastAPI
@app.get("/predict/{id_client}")
def predict(id_client: int):
    # Extraction des données associées à l'identifiant SK_ID_CURR sélectionné
    data_client = df.loc[df["SK_ID_CURR"] == id_client]

    # Préparation des données pour la prédiction
    X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba_1"], axis=1).values

    # Prédiction et probabilité 
    prediction = model.predict(X)
    pred_proba = model.predict_proba(X) 
    proba_paiment = pred_proba[0][1]
    proba_defaut_paiment = pred_proba[0][0] #1 - proba_paiment 

    return {"probaClasse1": proba_paiment, "probaClasse0": proba_defaut_paiment, "prediction": int(prediction[0]) }


#------------------------------------------------------


#Pour lancer l'API vous devez installer les dépendences en executant la commande suivante dans un terminal de préférence
#anaconda prompt: pip install -r requirements.txt
# puis la commande suivante: uvicorn fichier_FastApi:app --reload
