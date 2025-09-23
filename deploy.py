import joblib
from flask import Flask, request, jsonify
from utils import process_data
import traceback
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
from aliyun.log import LogClient, PutLogsRequest, LogItem, LogClient

app = Flask(__name__)


def init(model_path="models/best_model.pkl"):
    global model, features, today, access_key_id, access_key_secret, log_client
    print("Initializing...")
    features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    today = datetime.now()

      # 请替换为你的SLS项目所在地域的Endpoi
    log_client = None
    access_key_id = None
    access_key_secret = None
      # 你的本地日志文件路径
   

    print("Loading model...")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading model:", str(e))
        print(traceback.format_exc())
        raise

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        date_format = today.strftime("%Y-%m-%d")
        input_path = data.get('input_path') if 'input_path' in data else f'inference_data/{date_format}/datatest.txt'
        X, _ = process_data(input_path, features, None)

        predictions = model.predict(X)
    
        if 'output_path' in data:
            pd.DataFrame({'Occupancy_prediction': predictions}).to_csv(data['output_path'], index=False)
        
        response = {
            'status': 'success'
        }
        print("Prediction successful.")
        return jsonify(response), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return jsonify({'error': str(e)}), 500

@app.route('/model_check', methods=['POST'])
def model_check():
    print(f'Received model check request on: ')
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        days = data.get('days') if 'days' in data else 1
        cloud_log_info = data.get('cloud_log_info') if 'cloud_log_info' in data else None

        for d in range(days):
            date_format = (today - timedelta(days=d)).strftime("%Y-%m-%d")
            print(f"Checking model based on data from {date_format} ||")
            input_path = f'inference_data/{date_format}/datatest_real.txt'
            X, y = process_data(input_path, features)
            y_pred = model.predict(X)
            model_result = {
                'accuracy': accuracy_score(y, y_pred),
                'f1_score': f1_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'precision': precision_score(y, y_pred, average='weighted')
            }
            print(f"Model result on {date_format}: {model_result}")
            
            if cloud_log_info:
                log_items = []
                endpoint = cloud_log_info.get('endpoint')
                project_name = cloud_log_info.get('project_name')
                logstore_name = cloud_log_info.get('logstore_name')
                log_client = LogClient(endpoint, access_key_id, access_key_secret)
                
                log_item = LogItem()
                log_item.set_time(int(datetime.now().timestamp()))  # 设置时间戳为当前时间
                # 将整行日志作为'message'字段的内容
                log_item.set_contents([('accuracy', '0.91')]) 
        
                log_items.append(log_item)
                '''                
                for metric, value in model_result.items():
                    log_item = LogItem()
                    log_item.set_time(int(time.time()))  # 设置时间戳为当前时间
                    log_item.set_contents([(metric, value)]) 
                    log_items.append(log_item)
                ''' 
                log_client = LogClient(endpoint, access_key_id, access_key_secret)
                log_request = PutLogsRequest(project_name, logstore_name, '', '', log_items)
                log_response = log_client.put_logs(log_request)
                print(log_response)
                '''
                if log_response.get_status() == 200:
                    print(f"Log sent to cloud successfully.")
                else:
                    print(f"Error sending log to cloud: {log_response.get_status()}")
                '''
        response = {
            'status': 'success'
        }
        print("Model check finished.")
        return jsonify(response), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return jsonify({'error': str(e)}), 500

@app.route('/health_check', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None}), 200


if __name__ == '__main__':
    init()
    print("Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)