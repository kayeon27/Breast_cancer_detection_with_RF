import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data():
    df = pd.read_csv('./Data/data.csv', delimiter=';')
    #print("  *******************  ",df.columns.tolist()," ****************     ")
    #supprimer la colonne Id
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    #convertir les colonnes de type object en type float
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    return df


def handle_missing(df, strategy ='median'):
    #imputer la medaine aux valeurs manquantes
    imputer = SimpleImputer(strategy = strategy)
    df[df.columns] = imputer.fit_transform(df)
    return df


def preprocessing_data( test_size : float = 0.2, random_state : int = 42):

    #chargement et netoyyage des donnees
    df = load_data()
    df = handle_missing(df)

    #separer les donnees X features et y target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    #separer les donnees en train et test
    X_train, X_test ,y_train,y_test = train_test_split(X,y, test_size= test_size, random_state=random_state, stratify= y)

    #standartiser les données
    scaler =  StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train , X_test, y_train, y_test, df, X_train_scaled,X_test_scaled
