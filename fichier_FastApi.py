from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

# Configuration des options de CORS
origins = [
    "http://localhost:8501",  # autoriser les demandes provenant de Streamlit en local
    "https://streamlit-app-p7.onrender.com", # autoriser les demandes provenant de Streamlit en ligne
]

# Activation du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle LGBMClassifier pré-entraîné et enregistré
model = joblib.load('LGBM_best_model.joblib')

# Chargement des données
df = pd.read_csv('df_dash_10.csv')

# Définition du schéma de la requête
class PredictionRequest(BaseModel):
    id_client: int

# Définition du schéma de la réponse
class PredictionResponse(BaseModel):
    proba: float

# Définition de la route de l'API pour la prédiction
@app.get("/predict/{id_client}", response_model=PredictionResponse)
def predict(id_client: int):
    # Extraction des données associées à l'identifiant SK_ID_CURR sélectionné
    data_client = df.loc[df["SK_ID_CURR"] == id_client]

    # Préparation des données pour la prédiction
    X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).values

    # Prédiction de la probabilité de défaut de paiement
    proba = model.predict_proba(X)[:, 1][0]

    # Retour de la probabilité de défaut de paiement
    return {"proba": proba}

#Pour lancer l'API vous devez installer les dépendences en executant la commande suivante dans un terminal de préférence
#anaconda prompt: pip install -r requirements.txt
# puis la commande suivante: uvicorn fichier_fastapi:app --reload
