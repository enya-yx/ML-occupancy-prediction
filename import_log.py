import time
from aliyun.log import LogClient, PutLogsRequest, LogItem, LogClient

endpoint = 'cn-shanghai.log.aliyuncs.com'  # 请替换为你的SLS项目所在地域的Endpoint
access_key_id = ''
access_key_secret = ''
project_name = 'yx-occupancy'
logstore_name = 'occupancy-eas-log'
log_file_path = 'logs/log09201135.log'  # 你的本地日志文件路径
log_client = LogClient(endpoint, access_key_id, access_key_secret)

log_items = [] 
with open(log_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            print(line)
            log_item = LogItem()
            log_item.set_time(int(time.time()))  # 设置时间戳为当前时间
            # 将整行日志作为'message'字段的内容
            log_item.set_contents([('message', line)]) 
            # 你也可以在这里用正则表达式解析行内容，设置多个字段
            # log_item.set_contents([('level', 'ERROR'), ('message', line)])
            log_items.append(log_item)

client = LogClient(endpoint, access_key_id, access_key_secret)
 # 创建日志库，设置分区数和存储容量
# 创建请求并发送
request = PutLogsRequest(project_name, logstore_name, '', '', log_items)
response = client.put_logs(request)
print('Upload completed. Request ID:', response.get_request_id())