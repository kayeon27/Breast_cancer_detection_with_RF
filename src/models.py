import os
from scipy.stats import randint

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from src.data_preprocessing import preprocessing_data

#Construire un modèle de forêt aléatoire
def build_model_rf(
        n_estimators : int = 100,
        random_state : int = 42,
        **kwargs
)->RandomForestClassifier:
    #initialisation du modèle
    params = dict(n_estimators = n_estimators, random_state = random_state)
    params.update(kwargs)
    return RandomForestClassifier(**params)




def train_model(model, X_train, y_train ,model_path = 'src/random_forest_model.pkl' ):
    # Entraînement du modèle
    model.fit(X_train, y_train)

    #Création du repertoire de sauvegarde
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Sauvegarde du modèle
    joblib.dump(model, model_path)
    return model



def cross_validation (
        model,
        X,
        y,
        n_splits : int = 5,
        random_state : int = 42,
       
):
 
    #Kfold stratifié et validation croisée
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring =  ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(
        estimator= model ,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring
    )
    # Affichage des résultats
    for metric in scoring:
        scores = results[f'test_{metric}']
        print(f"{metric.capitalize():<10}:"
              f"{scores.mean():.3f} ± {scores.std():.3f}")
        


#optimiser les hyperparametres du modèle
def tune_random_forest(
        model,
        X,
        y,
        param_dist= None,
        n_iter : int = 10,
        cv_splits : int = 5,
        random_state : int = 42,
        model_path : str = 'src/best_random_forest.pkl',
):
    if param_dist is None:
        param_dist = {
            'n_estimators': randint(100,500),
            'max_depth': [None]+list(randint(2, 50).rvs(size=10)),
            'min_samples_split': randint(2, 30),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', 0.5, 0.75],
            'bootstrap': [True, False]
        }


    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    # Perform random search 
    search = RandomizedSearchCV (
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1', 
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
        )
    # Fit the search to the data
    search.fit(X.values if hasattr(X, 'values') else X,
                y.values if hasattr(y, 'values') else y)
    
    # Print the best parameters and score
    best_model = search.best_estimator_
    print("\n Meilleurs hyperparametres :")
    for k, v in search.best_params_.items():
        print(f"  {k:20s}: {v}")

    print(f"Meilleur score f1 en CV : {search.best_score_:.3f}")

    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True) 
    joblib.dump(best_model, model_path)
    print(f"Meilleur modèle enregistré sous {model_path}")
    return best_model