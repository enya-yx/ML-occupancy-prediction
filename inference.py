import requests
import json


class Client:
    def __init__(self, url, token=None):
        self.url = url
        self.token = token
        self.headers = {
            "Authorization": self.token,
            "Content-Type": "application/json"
        } if self.token else {
            "Content-Type": "application/json"
        }

    def predict(self, data):
        #urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)      
        response = requests.post(
            f"{self.url}/predict", json=data,headers=self.headers
        )

        return response

    def model_check(self, data):
        response = requests.post(
            f"{self.url}/model_check", json = data, headers = self.headers
        ) 
        return response  #print(response.json())

if __name__ == '__main__':
    #SERVICE_URL = "http://1381793422686229.cn-shanghai.pai-eas.aliyuncs.com/api/predict/occupancy_sklearn_clone6"
    #token = "OWI4Njc2YjkwMWFiN2VkNTU4YWUyZWIzODE4NjE3OWQyMjE0NGJhZA=="
    #client = Client(SERVICE_URL, token)
    client = Client("http://localhost:5000/")
    '''
    data = {
        'input_path': 'occupancy_data/datatest.txt',
        'output_path': 'occupancy_data/output.txt'
    }
    response = client.predict(data)
    print("Predict Result Status Code:", response.status_code)
    '''
    data2 = {
        'days': 1,
        'cloud_log_info': {
            'project_name': 'yx-occupancy',
            'logstore_name': 'occupancy-eas-log',
            'endpoint': 'cn-shanghai.log.aliyuncs.com'
        }
    }
    response2 = client.model_check(data2)
    print("Model Check Result Status Code:", response2.status_code)
    #print("Response:", response.json())