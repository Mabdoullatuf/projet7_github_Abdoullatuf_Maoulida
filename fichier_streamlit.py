import streamlit as st
import pandas as pd
import numpy as np 
import requests
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests
import pickle
import shap
shap.initjs()

#___________ Paramètres de la page_______________
# Interface utilisateur avec Streamlit
st.set_page_config(page_title="Analyse locale et globale de vos clients", page_icon=":money_with_wings:", layout="centered")
st.title("Analyse locale et globale de vos clients :money_with_wings:")
page_title = "Analyse locale et globale de vos clients"
page_icon = ":money_with_wings:"
layout = "centered"

# Charger les données
df = pd.read_csv('df_dash.csv')


# Charger le modèle LGBMClassifier pré-entraîné et enregistré
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)


def is_online_api_available(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except:
        return False
    
    
# Définition de la fonction pour effectuer la prédiction

def get_prediction(id_client):
    
    # URLs 
    local_url = f"http://127.0.0.1:8000/predict/{id_client}"
    online_url = f"https://fast-api-dashboard-final.onrender.com/predict/{id_client}"
    
    # Check if online API is available
    if is_online_api_available(online_url):
        response = requests.get(online_url)
    else:
        response = requests.get(local_url)
    
    prediction_data = response.json()
    return prediction_data





# Sélection de l'identifiant SK_ID_CURR à partir d'un menu déroulant
id_client = st.selectbox("Sélectionner l'identifiant du client", df["SK_ID_CURR"])

# Extraction des données associées à l'identifiant SK_ID_CURR sélectionné
data_client = df.loc[df["SK_ID_CURR"] == id_client]

# Afficher les infos du client sélectionné
st.dataframe(data_client)

# Obtenir la prédiction
prediction_data = get_prediction(id_client)
proba_defaut_paiment = prediction_data["probaClasse0"]
proba_paiment = prediction_data["probaClasse1"]
#prediction = prediction_data["prediction"]

# Continuons à construire notre tableau de bord à partir d'ici
# ...

# #___________ Paramètres de la page_______________
# page_title = "Analyse locale et globale de vos clients"
# page_icon = ":money_with_wings:"
# layout = "centered"

if proba_paiment >= 0.52:
    st.write('<p style="color: green; font-weight: bold; font-size: 24px;">Le client {} est éligible à un prêt avec une probabilité de paiement de {}%.</p>'.format(id_client, round(proba_paiment*100, 2)), unsafe_allow_html=True)
else:
    st.write('<p style="color: red; font-weight: bold;font-size: 24px;">Le client {} n\'est pas éligible à un prêt. Sa probabilité de paiement est faible, elle est de {}%. Le seuil optimal est de 52%.</p>'.format(id_client, round(proba_paiment*100, 2)), unsafe_allow_html=True)
    
    
# Préparation des données pour shap, la bilinéarité et éventuellement les prédictions (seulement ici on récupere les predictions
# à partir de l'app fastapi 
X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba_1"], axis=1).values
    
# Entraînement de shap sur le train set
bar_explainer = shap.Explainer(model, X)
bar_values = bar_explainer(X, check_additivity=False)    


def interpretabilite():
    ''' Affiche l'interpretabilite du modèle
    '''
    html_interpretabilite="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: white; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:white; color:Black;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      SHAP Values
                  </h3>
            </div>
        </div>
        """

    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.checkbox("Interpretabilité du modèle"):     
        
        st.markdown(html_interpretabilite, unsafe_allow_html=True)

        with st.spinner('**Affiche l\'interpretabilité du modèle...**'):                 
                       
            with st.expander('interpretabilité du modèle',
                              expanded=True):
                
                client_index = data_client.index.item()
                X_shap = X_shap = df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba_1"], axis=1)
                X_courant = X_shap.iloc[client_index]
                X_courant_array = X_courant.values.reshape(1, -1)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_courant_array)

               
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    

                    # BarPlot du client courant
                    shap.summary_plot(shap_values, X_shap, plot_type='bar')
                    
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()

                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values[1],
                                    X_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                    
st.subheader('Interpretabilité du modèle : Quelles variables sont les plus importantes ?')
interpretabilite()



#________Analyse_comparative_____________________

st.subheader('Analyse comparative entre le clients courant et les classes :')

list_features_1 = df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba_1"], axis=1).columns
feature_1 = st.selectbox('Selectionnez la feature à comparer :',list_features_1)

X_validé = df.loc[df['prediction']==1]
X_val = X_validé[feature_1].mean()

X_refusé = df.loc[df['prediction']==0]
X_ref = X_refusé[feature_1].mean()

X_client = data_client[feature_1].values.item()

clients = pd.DataFrame([["Moyenne_des_clients_validé", X_val], ["Moyenne_des_client_refusé", X_ref], ["Client_courant", X_client]], columns=["Clients","Valeur"])

fig = px.bar(clients, x='Clients', y=["Valeur"], barmode='group', height=400)

st.plotly_chart(fig)

#______SHAP_Global_____________________
def shap_global_analysis(model, X):
    st.subheader('Analyse SHAP Globale')
    
    # Calcul des valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Convertir shap_values en une liste de tableaux numpy
    shap_values = [np.array(sv) for sv in shap_values]

    # S'assurer que X est un DataFrame pandas avec des noms de colonnes
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba_1"], axis=1).columns)

    # Création du summary plot
    st.set_option('deprecation.showPyplotGlobalUse', False) # pour supprimer les warnings Streamlit liés à l'utilisation de pyplot
    plt.figure(figsize=(10,5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot()
    


    # Appel de la fonction
if st.checkbox("Afficher l'analyse globale SHAP"):
    #X = df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba_1"], axis=1)
    shap_global_analysis(model, X)




#________Analyse_bivariée_____________________

list_features_2 = df.drop(["SK_ID_CURR", "prediction", "proba_1"], axis=1).columns
feature_2 = st.selectbox('Selectionnez la première feature :',list_features_2)
feature_3 = st.selectbox('Selectionnez la deuxième feature :',list_features_2)

x=df[feature_2]
y=df[feature_3]

plot = px.scatter(x=x, y=y)

client_point = plot.add_trace(go.Scatter(x=data_client[feature_2].values, y=data_client[feature_3].values, mode = 'markers', marker_symbol = 'star', marker_size = 15))

st.plotly_chart(plot)

    

st.write("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #F5F5F5; padding: 10px; text-align: center;">
        Abdoullatuf Maoulida
    </div>
""", unsafe_allow_html=True)

#--------------------------------------------------------------------------





#Pour lancer l'API vous devez installer les dépendences en executant la commande suivante dans un terminal de préférence
#anaconda prompt: pip install -r requirements.txt
# puis la commande suivante (après avoir lancé le fichier_FastApi): streamlit run fichier_streamlit.py

