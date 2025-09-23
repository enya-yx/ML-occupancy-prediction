import argparse
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
import mlflow
import mlflow.sklearn
from datetime import datetime

def load_data(train_path, test_path, features_path=None):
    
    features = []
    if features_path is None:
        features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    else :
        features = pd.read_csv(features_path, header=None)[0].tolist()

    X_train, y_train = process_data(train_path, features, 'Occupancy')
    X_test, y_test = process_data(test_path, features, 'Occupancy')

    return X_train, y_train, X_test, y_test

def train_sklearn_models(X_train, y_train, X_test, y_test, exp_id, hyper_params):
    rs = 42 if hyper_params and 'random_state' not in hyper_params else hyper_params['random_state']
    es = 100 if hyper_params and 'n_estimators' not in hyper_params else hyper_params['n_estimators']
    '''
    TODO: more hyper parameters and models to try
    '''
    models = {
        'Logistic Regression': LogisticRegression(random_state=rs),
        'Random Forest': RandomForestClassifier(n_estimators=es, random_state=rs),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=es, random_state=rs)
    }
    results = {}
    best_accuracy = 0
    best_model = None
    print("Experiment ID: ", exp_id)
    with mlflow.start_run(experiment_id=exp_id):
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
                'predictions': y_pred,
                'f1_score': test_f1
            }
            if accuracy > best_accuracy:
                best_model = model
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_model_name", best_model.name)
        mlflow.log_param("hyper_params_random_state", rs)
        mlflow.log_param("hyper_params_n_estimators", es)
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.sklearn.log_model(best_model, f"best_model_{exp_id}")
        mlflow.log_artifacts('models')
        print(f"best model: {best_model.name}, accuracy: {best_accuracy:.4f}")
    model_path = 'models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f'best model saved to {model_path}')

    return results,best_model

def main():
    time = datetime.now()
    exp = mlflow.set_experiment(f'occupancy_prediction_{time.strftime("%Y%m%d_%H%M%S")}')
    train_path = argparse.ArgumentParser().add_argument('--train_path', type=str, default='occupancy_data/datatraining.txt')
    test_path = argparse.ArgumentParser().add_argument('--test_path', type=str, default='occupancy_data/datatest.txt')
    features_path = argparse.ArgumentParser().add_argument('--features_path', type=str, default=None)
    hyper_params = argparse.ArgumentParser().add_argument('--hyper_params', type=dict, default={'random_state': 42, 'n_estimators': 100})
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, features_path)
    
    print(X_train[0:2])
    
    all_results,best_model = train_sklearn_models(X_train, y_train, X_test, y_test, exp.experiment_id, hyper_params)
    #best_model_name = max(all_results.items(), key=lambda x: x[1]['accuracy'])[0]
    #best_accuracy = all_results[best_model_name]['accuracy']
    #print(all_results)
    #print(f"best model: {best_model.best_model_name}, accuracy: {best_model.accuracy:.4f}")
    
    #return all_results

if __name__ == '__main__':
    main()