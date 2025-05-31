import os
import joblib
import pandas as pd
import streamlit as st
from typing import List
import numpy as np



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#charger le model
model_path = os.path.join(BASE_DIR,"random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR,"scaler.pkl")
data_path = os.path.join(BASE_DIR, "../Data/data.csv")

st.write("BASE_DIR:", BASE_DIR)
st.write("MODEL_PATH:", model_path)
st.write("SCALER_PATH:", scaler_path)
st.write("DATA_PATH:", data_path)



model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Charger le scaler

#Charger la liste des features dans l'ordre
df = pd.read_csv(data_path, sep=';')
FEATURE_NAMES = [c for c in df.columns if c not in ("id", "diagnosis") ]


#Titre et description
st.title("Breast Cancer Prediction ")
st.markdown("Entrez les **30 caractéristiques** du patient pour savoir s'il est prédit comme **malade** ou **sain**."
)

#Choix de l'input
mode = st.radio("Mode d'entrée" , ("Manuel","fichier CSV"), key= "mode_selector")
if mode == "Manuel":
    st.subheader("Entrée manuelle des caractéristiques")
    #afficher un slider ou input numerique pour chaque feature 
    inputs ={}
    for idx, feature in enumerate(FEATURE_NAMES):
          #ajoute une clé unique/feature
        inputs[feature]= st.number_input(
            label= feature, 
            value = 0.0, 
            format = "%.4f",
            key= f"input_{idx}"
            )

        #quand l'utilisateur clique
    if st.button("Predire (manuel)", key = "btn_manual_pred"):
            #construire le tableau et le scaler
            x = np.array([inputs[f] for f in FEATURE_NAMES]).reshape(1, -1)
            x_scaled = scaler.transform(x)
            
            #faire la prédiction
            proba = model.predict_proba(x_scaled)[0,1]
            pred = model.predict(x_scaled)[0]
            st.success(f"Prédiction : **{'Malade' if pred ==1 else 'Sain'}**")
            st.write(f"Probabilité de malignité : {proba:.3f}")

else: 
    #mode fichier CSV
    st.subheader("Chargement d'un fichier CSV")
    uploaded_file = st.file_uploader(
        label="Chargez un CSV avec les colonnes suivantes : ", 
        type = ["csv"], 
        key = "uploader_csv"
        )
    st.caption("colonnes attendus : "+ ",".join(FEATURE_NAMES))
    if uploaded_file is not None:
        #lire le fichier csv
        df = pd.read_csv(uploaded_file, sep=';' ,engine='python')
        #verifier les colonnes
        missing = set(FEATURE_NAMES) - set(df.columns)
        if missing: 
            st.error(f"Colonnes manquantes : {missing}")
        else: 
            if st.button("Prédire batch" , key= "btn_batch_pred"):
                X = df[FEATURE_NAMES].values
                #scaler les données
                X_scaled = scaler.transform(X)
                #faire la prédiction
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1]
                df["predicted_class"] = preds
                df["malignant_probability"] = probs
                st.dataframe(df)
                st.download_button(
                    label="Télécharger les résultats CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="predictions.csv",
                    mime = "text/csv",
                    key= "btn_download_csv"
                )
