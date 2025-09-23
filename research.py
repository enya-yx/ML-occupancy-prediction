from pyexpat import features
import joblib
import pandas as pd
from scipy.stats import f
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from utils import process_data

def load_data(train_path, test_path, features_path=None):
    
    features = []
    if features_path is None:
        features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    else :
        features = pd.read_csv(features_path, header=None)[0].tolist()

    X_train, y_train = process_data(train_path, features, 'Occupancy')
    X_test, y_test = process_data(test_path, features, 'Occupancy')

    return X_train, y_train, X_test, y_test

def train_sklearn_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    best_accuracy = 0
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'{name} accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        if accuracy > best_accuracy:
            best_model = model
    model_path = 'models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f'best model saved to {model_path}')

    return results,best_model

def main():
    train_path = 'occupancy_data/datatraining.txt'
    test_path = 'occupancy_data/datatest.txt'
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    print(X_train[0:2])
    all_results,best_model = train_sklearn_models(X_train, y_train, X_test, y_test)
    #best_model_name = max(all_results.items(), key=lambda x: x[1]['accuracy'])[0]
    #best_accuracy = all_results[best_model_name]['accuracy']
    #print(all_results)
    #print(f"best model: {best_model.best_model_name}, accuracy: {best_model.accuracy:.4f}")
    
    #return all_results

if __name__ == '__main__':
    main()