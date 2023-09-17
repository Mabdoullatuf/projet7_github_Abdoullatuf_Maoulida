# projet7_model_scoring_Abdoullatuf_Maoulida
Projet de Prédiction avec LGBMClassifier

Description:
Ce projet a pour objectif de développer un modèle prédictif basé sur le LGBMClassifier. À travers un processus rigoureux de sélection de fonctionnalités, 
d'optimisation d'hyperparamètres et de validation avec des métriques métier spécifiques, nous avons cherché à créer un modèle robuste et performant pour notre cas d'utilisation.

Caractéristiques principales
Sélection de Fonctionnalités : Utilisation de RFECV pour identifier et conserver les caractéristiques les plus pertinentes.

Optimisation d'Hyperparamètres : Adoption de Hyperopt pour le réglage des paramètres du LGBMClassifier, avec pour objectif d'optimiser une métrique métier définie.

Dashboard Intuitif : Grâce à Streamlit et FastAPI, un dashboard interactif a été mis en place pour une visualisation facile et une utilisation en temps réel du modèle.

Détection de Data Drift : Mise en œuvre d'une solution avec Evidently pour suivre et alerter sur les possibles dérives des données.

Déploiement : Le modèle et le dashboard ont été déployés sur Render.com (https://fast-api-dashboard-final.onrender.com  et  https://streamlit-app-p7.onrender.com) pour une accessibilité universelle.

Limitations et Défis
Bien que le LGBMClassifier soit performant, il présente certaines limitations, notamment en matière d'interprétabilité et de complexité d'hyperparamétrage.
La détection du data drift, bien que cruciale, nécessite une maintenance régulière pour assurer sa pertinence.

Prochaines étapes:
Des améliorations sont déjà envisagées pour rendre le modèle plus robuste, plus efficace et plus convivial. Cela inclut l'exploration d'autres modèles, l'ajout de fonctionnalités d'interprétabilité et 
l'intégration de retours d'information en boucle fermée pour une amélioration continue.

Installation et utilisation
Voir le fichier "requirements.txt" pour les versions des bibliothèques et python utilisées.

Contact
Pour toute question ou feedback, n'hésitez pas à me contacter à a.ellatuf@gmail.com.
