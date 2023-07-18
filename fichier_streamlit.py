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
model = joblib.load('final_model.joblib')

# Chargement des données
df = pd.read_csv('df_dash.csv')
<<<<<<< HEAD

#___________ Paramètres de la page_______________
page_title = "Analyse locale et globale de vos clients"
page_icon = ":money_with_wings:"
layout = "centered"


#___________Affichage_infos_client_________________________

# Interface utilisateur avec Streamlit
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

st.write("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #F5F5F5; padding: 10px; text-align: center;">
        Abdoullatuf Maoulida
    </div>
""", unsafe_allow_html=True)



#--------------------------------------------------------------------------


# URL de l'API FastAPI
#url = "http://localhost:8000/predict/"
<<<<<<< HEAD
url = "https://fast-api-dashboard.onrender.com/predict/"
=======
url = "https://fast-api-dashboard.onrender.com/predict"
>>>>>>> 379d8f583470ce4da8dc8b2d8243952f2361b024


# Définition de la fonction pour effectuer la prédiction
def get_prediction(id_client):
<<<<<<< HEAD
    response = requests.get(url + str(id_client))
=======
    response = requests.post(url + str(id_client))
>>>>>>> 379d8f583470ce4da8dc8b2d8243952f2361b024
    proba = response.json()['proba']
    return proba


# Sélection de l'identifiant SK_ID_CURR à partir d'un menu déroulant
id_client = st.selectbox("Sélectionner l'identifiant du client", df["SK_ID_CURR"])

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
pred_proba = model.predict_proba(X) 
proba_defaut_paiment = pred_proba[0][0]
proba_paiment = 1 - proba_defaut_paiment




if proba_defaut_paiment < 0.55:
    st.write('<p style="color: green; font-weight: bold; font-size: 24px;">Le client {} est éligible à un prêt avec une probabilité de payement de {}%.</p>'.format(id_client, round(proba_paiment*100, 2)), unsafe_allow_html=True)
else:
    st.write('<p style="color: red; font-weight: bold;font-size: 24px;">Le client {} n\'est pas éligible à un prêt. Sa probabilité de payement est faible, elle est de {}%.</p>'.format(id_client, round(proba_paiment*100, 2)), unsafe_allow_html=True)


#________Feature_importance_locale________________________


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

#________Analyse_comparative_____________________

st.subheader('Analyse comparative entre le clients courant et les classes :')

list_features = df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).columns
feature_1 = st.selectbox('Selectionnez la feature à comparer :',list_features)

X_validé = df.loc[df['prediction']==1]
X_val = X_validé[feature_1].mean()

X_refusé = df.loc[df['prediction']==0]
X_ref = X_refusé[feature_1].mean()

X_client = data_client[feature_1].values.item()

clients = pd.DataFrame([["Moyenne_des_clients_validé", X_val], ["Moyenne_des_client_refusé", X_ref], ["Client_courant", X_client]], columns=["Clients","Valeur"])

fig = px.bar(clients, x='Clients', y=["Valeur"], barmode='group', height=400)

st.plotly_chart(fig)

#______SHAP_Global_____________________





#________Analyse_bivariée_____________________

feature_2 = st.selectbox('Selectionnez la première feature :',list_features)
feature_3 = st.selectbox('Selectionnez la deuxième feature :',list_features)

x=df[feature_2]
y=df[feature_3]

plot = px.scatter(x=x, y=y)

client_point = plot.add_trace(go.Scatter(x=data_client[feature_2].values, y=data_client[feature_3].values, mode = 'markers', marker_symbol = 'star', marker_size = 15))

st.plotly_chart(plot)


=======
>>>>>>> efdd1c2b62ea2092bcc5245e49fd3218be6ef87b

#___________ Paramètres de la page_______________
page_title = "Analyse locale et globale de vos clients"
page_icon = ":money_with_wings:"
layout = "centered"


#___________Affichage_infos_client_________________________

# Interface utilisateur avec Streamlit
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

st.write("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #F5F5F5; padding: 10px; text-align: center;">
        Abdoullatuf Maoulida
    </div>
""", unsafe_allow_html=True)



#--------------------------------------------------------------------------


# URL de l'API FastAPI
#url = "http://localhost:8000/predict/"
url = "https://fast-api-dashboard.onrender.com/predict/"


# Définition de la fonction pour effectuer la prédiction
def get_prediction(id_client):
    response = requests.get(url + str(id_client))
    proba = response.json()['proba']
    return proba


# Sélection de l'identifiant SK_ID_CURR à partir d'un menu déroulant
id_client = st.selectbox("Sélectionner l'identifiant du client", df["SK_ID_CURR"])

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
pred_proba = model.predict_proba(X) 
proba_defaut_paiment = pred_proba[0][0]
proba_paiment = 1 - proba_defaut_paiment




if proba_defaut_paiment < 0.55:
    st.write('<p style="color: green; font-weight: bold; font-size: 24px;">Le client {} est éligible à un prêt avec une probabilité de payement de {}%.</p>'.format(id_client, round(proba_paiment*100, 2)), unsafe_allow_html=True)
else:
    st.write('<p style="color: red; font-weight: bold;font-size: 24px;">Le client {} n\'est pas éligible à un prêt. Sa probabilité de payement est faible, elle est de {}%.</p>'.format(id_client, round(proba_paiment*100, 2)), unsafe_allow_html=True)


#________Feature_importance_locale________________________


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

#________Analyse_comparative_____________________

st.subheader('Analyse comparative entre le clients courant et les classes :')

list_features = df.drop(["SK_ID_CURR", "TARGET", "prediction", "proba"], axis=1).columns
feature_1 = st.selectbox('Selectionnez la feature à comparer :',list_features)

X_validé = df.loc[df['prediction']==1]
X_val = X_validé[feature_1].mean()

X_refusé = df.loc[df['prediction']==0]
X_ref = X_refusé[feature_1].mean()

X_client = data_client[feature_1].values.item()

clients = pd.DataFrame([["Moyenne_des_clients_validé", X_val], ["Moyenne_des_client_refusé", X_ref], ["Client_courant", X_client]], columns=["Clients","Valeur"])

fig = px.bar(clients, x='Clients', y=["Valeur"], barmode='group', height=400)

st.plotly_chart(fig)

#______SHAP_Global_____________________





#________Analyse_bivariée_____________________

feature_2 = st.selectbox('Selectionnez la première feature :',list_features)
feature_3 = st.selectbox('Selectionnez la deuxième feature :',list_features)

x=df[feature_2]
y=df[feature_3]

plot = px.scatter(x=x, y=y)

client_point = plot.add_trace(go.Scatter(x=data_client[feature_2].values, y=data_client[feature_3].values, mode = 'markers', marker_symbol = 'star', marker_size = 15))

st.plotly_chart(plot)









#Pour lancer l'API vous devez installer les dépendences en executant la commande suivante dans un terminal de préférence
#anaconda prompt: pip install -r requirements.txt
# puis la commande suivante (après avoir lancé le fichier_FastApi): streamlit run fichier_streamlit.py

