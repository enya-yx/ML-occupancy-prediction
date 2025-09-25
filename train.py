import argparse
from os import name
from pyexpat import features
import joblib
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from utils import process_data
from datetime import date, datetime
from aliyun.log import LogClient, PutLogsRequest
from aliyun.log.logitem import LogItem
import os
'''
endpoint = 'cn-shanghai.log.aliyuncs.com'  
access_key_id = ''
access_key_secret = ''
project_name = 'yx-occupancy'
logstore_name = 'occupancy-eas-log'
'''
endpoint = os.environ.get('ALIYUN_LOG_ENDPOINT') 
access_key_id = os.environ.get('ALIYUN_LOG_ACCESS_KEY_ID') 
access_key_secret = os.environ.get('ALIYUN_LOG_ACCESS_KEY_SECRET') 
project_name = os.environ.get('ALIYUN_LOG_PROJECT_NAME') 
logstore_name = os.environ.get('ALIYUN_LOG_LOGSTORE_NAME') 

# Put logs into training logstore to track the training process
log_client = LogClient(endpoint, access_key_id, access_key_secret)

def load_data(train_path, test_path, features_path=None):
    '''
    Load data from train_path and test_path, and process the data using process_data function.
    If features_path is not None, load features from features_path.
    '''
    print(f"load data from {train_path} and {test_path}")
    X_train, y_train, X_test, y_test = None, None, None, None
    try:
        features = []
        if features_path is None:
            features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        else :
            features = pd.read_csv(features_path, header=None)[0].tolist()
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        X_train, y_train = process_data(train_data, features, 'Occupancy')
        X_test, y_test = process_data(test_data, features, 'Occupancy')
        print(f"load data successfully. train data path: {train_path}, test data path: {test_path}, features: {features}")
    except Exception as e:
        print(f"load data failed. error: {e}")
    return X_train, y_train, X_test, y_test

def train_sklearn_models(X_train, y_train, X_test, y_test, hyper_params_args: dict):
    '''
    Train sklearn models with hyper_params.
    Return the results and the best model.
    Save new models to models/{today}today/*.pkl. (aliyun OSS)
    Save best model to models/best_model.pkl.
    Save log to {today}_train.log. (aliyun SLS)
    '''
    rs = hyper_params_args['random_state'] if hyper_params_args and hasattr(hyper_params_args, 'random_state') else 42
    es = hyper_params_args['n_estimators'] if hyper_params_args and hasattr(hyper_params_args, 'n_estimators') else 100
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
    best_model, best_model_name = None, None
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    exp_id = now.strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = date_str + '_train.log'
    print("Experiment ID: ", exp_id)
    log_items = []
    # Log hyper parameters
    log_item = LogItem()
    log_item.set_time(int(datetime.now().timestamp()))
    log_item.set_contents([('exp_id', exp_id), ('n_estimators', f'{es}'), ('random_state', f'{rs}')])
    log_items.append(log_item)

    # Train each models and get the best model
    directory_path = f"models/{date_str}"
    os.makedirs(directory_path, exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        print(f'{name} accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
        joblib.dump(model, f'{directory_path}/{name}.pkl')
        print(f'{name} model saved to {directory_path}/{name}.pkl')
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': test_f1,
            'recall_score': recall
        }
        log_item = LogItem()
        log_item.set_time(int(datetime.now().timestamp()))
        log_item.set_contents([('exp_id', exp_id), ('model', name), ('accuracy', f'{accuracy}'), ('f1_score', f'{test_f1}'), ('recall_score', f'{recall_score}')])
        log_items.append(log_item)
        if accuracy > best_accuracy:
            best_model = model
            best_model_name = name
            best_accuracy = accuracy
       
    print(f"best model: {best_model_name}, accuracy: {best_accuracy:.4f}")
    item = LogItem()
    item.set_time(int(datetime.now().timestamp()))
    item.set_contents([('exp_id', exp_id), ('best_model', best_model_name), ('best_accuracy', f'{best_accuracy}')])
    log_items.append(item)
    log_request = PutLogsRequest(project_name, logstore_name, '', '', log_items)
    try: 
        log_response = log_client.put_logs(log_request)   
        print(f"log saved to {log_file_path}")
    except Exception as e:
        print(f"PutLogs failed. error: {e}")

    model_path = 'models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f'best model saved to {model_path}')

    return results,best_model

def main():
    time = datetime.now()
    date_str = time.strftime('%Y-%m-%d')
    print(f"Start training at {date_str}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=f'train_data/{date_str}/datatraining.txt')
    parser.add_argument('--test_path', type=str, default=f'train_data/{date_str}/datatest.txt')
    parser.add_argument('--features_path', type=str, default=None)
    parser.add_argument('--hyper_params', type=dict, default=None)
    args = parser.parse_args()
    X_train, y_train, X_test, y_test = load_data(args.train_path, args.test_path, args.features_path)
        
    all_results,best_model = train_sklearn_models(X_train, y_train, X_test, y_test, args.hyper_params)

if __name__ == '__main__':
    main()