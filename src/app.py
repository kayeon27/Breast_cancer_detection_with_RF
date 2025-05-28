import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

#charger le model
model_path = "src/random_forest_model.pkl"
scaler_path = "src/scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Charger le scaler si utilisé

#Charger la liste des features dans l'ordre
df = pd.read_csv("Data/data.csv", sep=';')
feature_name = df.drop (columns=["id", "diagnosis"]).columns.tolist()

#definir le format de la requete
class predict_request(BaseModel):
    data: List[float] = Field(..., 
                                  description=f"Liste de {len(feature_name)} valeurs des features dans l'ordre {feature_name}")
    

#definir le format de la réponse
class predict_response(BaseModel):
    predicted_class  : int = Field(..., description="0 = saint, 1 = malade")
    malignant_probability : float = Field(..., description="probabilié d'etre malade")

#Initialiser FastApi
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API pour predire si un patient est malade (cancer du sein).",
    version="1.0"
)

@app.get("/", tags=["root"])
def read_root():
    return {
        "message": "Bienvenue ! Pour prédire, faites POST /predict. Voir /docs pour l’interface interactive."
    }


@app.post("/predict", response_model = predict_response)
def predict(req : predict_request):
    #verifier le nombre de features
    if len(req.data) != len(feature_name):
        raise HTTPException(
            status_code = 400,
            detail = f"le nombre de features doit etre {len(feature_name)}, mais {len(req.data)}a eté fourni."
        )
    
    #construire le tableau et le scaler
    x = np.array(req.data, dtype=float).reshape(1, -1)
    try:
        x_scaled = scaler.transform(x)
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de transformation des données: {e}"
        )
    
    #faire la prédiction
    proba = model.predict_proba(x_scaled)[0,1] # Probabilité d'être malade
    pred = int(model.predict(x_scaled)[0])  # Prédiction de la classe (0 ou 1)

    return predict_response(
        predicted_class = pred,
        malignant_probability = float(proba)
    )