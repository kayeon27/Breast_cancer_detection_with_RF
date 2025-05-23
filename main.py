import os
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
import joblib
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.test_prep import preprocess_prognostic  
from src.data_preprocessing import preprocessing_data
from src.models import build_model_rf, train_model, cross_validation, tune_random_forest

print("Running:", __file__)
print("   repertoire courant  :", os.getcwd())



if __name__ == "__main__":
    # Charger et pretraiter les données  
    X_train , X_test, y_train, y_test, df, X_train_scaled,X_test_scaled = preprocessing_data(test_size=0.2, random_state=42)
    
    #concatener les données pour la CV et le tuning
    X =  np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    #Construction du modele
    rf = build_model_rf(n_estimators=100, random_state=42)

    #entrainer le modèle
    trained_rf = train_model(rf, X_train, y_train)

    #Afficher les 5 premières lignes du jeu de données
    print(" ", df[:5])
    # Afficher les dimensions des données
    print(f"y_train shape: {y_train.shape}")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_test shape: {y_test.shape}")
    

    #Faire des prédictions sur les données de test   
    y_pred = trained_rf.predict(X_test)
    #Afficher les 5 premières prédictions
    print("Predictions:", y_pred[:5])


    # y_test = ground-truth, y_pred = prédictions du modèle
    print("Classification report:")
    print (classification_report (y_test, y_pred))

    print("matrice confusion")
    print(confusion_matrix(y_test, y_pred))


    #afficher la précision du modèle
    accuracy = trained_rf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    #cross_validation
    cross_validation(rf, X,y)

     #Affinage des meilleures hyperparametres du modele
    print("\n====Hyperparameter tuning===")
    best_model_path = "src/best_random_forest.pkl"
    best_rf = joblib.load(best_model_path)

    #Afficher les predictions du meilleur modèle
    y_pred_best = best_rf.predict(X_test.values if hasattr(X_test, 'values') else X_test)
    print("Best model predictions:", y_pred_best[:5])

    print("Best model classification report:")
    print(classification_report(y_test, y_pred_best))

    print("Best model confusion matrix")
    print(confusion_matrix(y_test, y_pred_best))

    

print("################################################################################")
print("############################# Évaluation du modèle sur le jeu Prognostic #############################")

# Chargement et prétraitement du jeu Prognostic
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_prognostic()

# Charger le meilleur modèle
best_model_path = "src/best_random_forest.pkl"
best_rf = joblib.load(best_model_path)

# Prédictions sur le jeu Prognostic
y_pred_prog = best_rf.predict(X_test_scaled)

# Évaluation
print("\n--- Évaluation du modèle optimisé sur le jeu Breast Cancer Wisconsin (Prognostic) ---")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_prog))
print("Accuracy :", accuracy_score(y_test, y_pred_prog))
print("Classification Report :\n", classification_report(y_test, y_pred_prog))
