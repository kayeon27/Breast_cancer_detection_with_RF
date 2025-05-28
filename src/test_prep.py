import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_prognostic(test_size=0.2, random_state=42):
    """
    Prépare les données Breast Cancer Wisconsin Prognostic pour la prédiction de récidive.
    """

    # Définir les noms des colonnes
    column_names = [
        'ID', 'Outcome',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
        'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
        'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    df = pd.read_csv("./Data/wpbc.data", header=None)

     #Vérifie le nombre de colonnes
    if df.shape[1] != 35:
        raise ValueError(f"Le fichier doit contenir 35 colonnes (ID, Outcome, 30 features, 3 autres). Actuellement : {df.shape[1]}")

    # Définir les noms de colonnes 
    df.columns = ["ID", "Outcome"] + [f"feature_{i}" for i in range(1, 31)] + ["extra1", "extra2", "extra3"]

    # Convertir la variable cible 
    df["Outcome"] = df["Outcome"].map({"N": 0, "R": 1})

    # Sélectionner uniquement les 30 features d'origine
    X = df[[f"feature_{i}" for i in range(1, 31)]]
    y = df["Outcome"]

    # Imputation
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Split
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train_scaled, X_test_scaled, y_train, y_test