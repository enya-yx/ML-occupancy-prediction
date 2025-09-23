from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def process_data(path: str,features: list[str], tar = 'Occupancy') -> pd.DataFrame:
    data = pd.read_csv(path)
    X = data[features] 
    y = data[tar] if tar in data.columns else None


    if 'HumidityRatio' in X.columns:
        X = X.assign(HumidityRatio = X['HumidityRatio'].round(5))
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    #X_train['Humidity'] = X_train['Humidity'].round(2)
    #X_test['Humidity'] = X_test['Humidity'].round(2)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # fi
    
    return X, y

