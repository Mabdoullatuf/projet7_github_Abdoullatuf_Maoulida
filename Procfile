#web: sh setup.sh && streamlit run fichier_streamli.py && uvicorn fichier_FastApi:app --host=0.0.0.0 --port=$PORT
web: uvicorn fichier_FastApi:app --host 0.0.0.0 --port $PORT
streamlit: streamlit run fichier_streamli.py