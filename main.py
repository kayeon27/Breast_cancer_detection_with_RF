import os

import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print("Running:", __file__)
print("   repertoire courant  :", os.getcwd())


from src.data_preprocessing import preprocessing_data
from src.models import build_model_rf, train_model, cross_validation, tune_random_forest

if __name__ == "__main__":
    # Load and preprocess the data 
    X_train , X_test, y_train, y_test, df, X_train_scaled,X_test_scaled = preprocessing_data(test_size=0.2, random_state=42)
    
    #concatener les données pour la CV et le tuning
    X =  np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    #Construction du modele
    rf = build_model_rf(n_estimators=100, random_state=42)

    #entrainer le modèle
    trained_rf = train_model(rf, X_train, y_train)

    # Print the first few rows of the data
    print(" ", df[:5])
    # Print the shapes of the datasets
    print(f"y_train shape: {y_train.shape}")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_test shape: {y_test.shape}")
    

    # Print the first few rows of the scaled data   
    y_pred = trained_rf.predict(X_test)
        # Print the predictions
    print("Predictions:", y_pred[:5])


    # y_test = ground-truth, y_pred = prédictions du modèle
    print("Classification report:")
    print (classification_report (y_test, y_pred))

    print("matrice confusion")
    print(confusion_matrix(y_test, y_pred))




    # Print the model's accuracy
    accuracy = trained_rf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    #cross_validation
    cross_validation(rf, X,y)

    #hyperparameter tuning
    print("\n====Hyperparameter tuning===")
    best_model = tune_random_forest(rf, X, y, n_iter=10, cv_splits=5)

      