from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def process_data(data: pd.DataFrame,features: list[str], tar = 'Occupancy') -> pd.DataFrame: 
    X = data[features] 
    y = data[tar] if tar in data.columns else None
    print(f"X shape: {X.shape}, y shape: {y.shape if y is not None else None}")

    if 'HumidityRatio' in X.columns:
        X = X.assign(HumidityRatio = X['HumidityRatio'].round(5))
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  
    
    return X, y
