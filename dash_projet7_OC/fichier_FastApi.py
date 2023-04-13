from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

# Chargement du modèle LGBMClassifier pré-entraîné et enregistré
model = joblib.load('LGBM_best_model.joblib')

# Chargement des données
df = pd.read_csv('df_dash_10.csv')

# Définition de la route de l'API
@app.get("/predict/{id_client}")
def predict(id_client: int):
    # Extraction des données associées à l'identifiant SK_ID_CURR sélectionné
    data_client = df.loc[df["SK_ID_CURR"] == id_client]

    # Préparation des données pour la prédiction
    X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).values
    #X = np.nan_to_num(X)

    # Prédiction de la probabilité de défaut de paiement
    proba = model.predict_proba(X)[:, 1][0]

    # Retour de la probabilité de défaut de paiement
    return {"probabilité de défaut de paiement": proba}

#Pour lancer l'API vous devez installer les dépendences en executant la commande suivante dans un terminal de préférence
#anaconda prompt: pip install -r requirements.txt
# puis la commande suivante: uvicorn fichier_fastapi:app --reload