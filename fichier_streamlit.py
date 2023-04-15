
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
import plotly.graph_objects as go
shap.initjs()


# Chargement du modèle LGBMClassifier pré-entraîné et enregistré
model = joblib.load('LGBM_best_model.joblib')

# Chargement des données
df = pd.read_csv('df_dash_10.csv')

#___________ Paramètres de la page_______________
page_title = "Analyse locale et globale de vos clients"
page_icon = ":money_with_wings:"
layout = "centered"


#___________Affichage_infos_client_________________________

# Interface utilisateur avec Streamlit
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)




# Sélection de l'identifiant SK_ID_CURR à partir d'un menu déroulant
id_client = st.selectbox("Sélectionner l'identifiant du client", df["SK_ID_CURR"])

#--------------------------------------------------------------------------
# Définition de l'URL de l'API FastAPI
url = "https://fast-api-dashboard-final.onrender.com/predict/{}".format(id_client)

# Envoi de la requête à l'API FastAPI
response = requests.get(url)

# Traitement de la réponse de l'API FastAPI
if response.status_code == 200:
    proba = response.json()["probabilité defaut de payement"]
    st.write(f"Probabilité défaut de payement : {proba:.2%}")
else:
    st.write("Erreur lors de l'appel à l'API FastAPI : {}".format(response.text))

#------------------------------------------




# Extraction des données associées à l'identifiant SK_ID_CURR sélectionné
data_client = df.loc[df["SK_ID_CURR"] == id_client]

#afficher les infos du client selectionné
st.dataframe(data_client)


# Préparation des données pour la prédiction
X = data_client.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).values
#X = np.nan_to_num(X)

# Prédiction et probabilité 
prediction = model.predict(X)
proba = model.predict_proba(X)[:, 1][0]


if prediction == 1:
    st.write('<p style="color: green; font-weight: bold; font-size: 24px;">Le client {} est éligible à un prêt avec une probabilité de payement de {}%.</p>'.format(id_client, round(proba*100, 2)), unsafe_allow_html=True)
else:
    st.write('<p style="color: red; font-weight: bold;font-size: 24px;">Le client {} n\'est pas éligible à un prêt. Sa probabilité de défaut de payement est de {}%.</p>'.format(id_client, round(proba*100, 2)), unsafe_allow_html=True)


#________Feature_importance_locale________________________

#display_f_importance = display_client.drop([

#explainer = shap.TreeExplainer(lgbm_model)
#shap_values = explainer.shap_values(display_client)
# Summary plot
#shap.plots.bar(shap_values, max_display=15)

#X_test_10_ = X_test_10.drop(['Unnamed: 0', 'index', 'prediction', 'probability'], axis=1)

 # Test set sans l'identifiant
#X_bar = X_test_10_.set_index('level_0')
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
                X_shap = X_shap = df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1)
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








#Pour lancer l'API vous devez installer les dépendences en executant la commande suivante dans un terminal de préférence
#anaconda prompt: pip install -r requirements.txt
# puis la commande suivante (après avoir lancé le fichier_FastApi): streamlit run fichier_streamlit.py
