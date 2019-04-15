import requests
import json

url = 'http://0.0.0.0:8500'
with open('test_128.json') as json_file:
    json_data = json.load(json_file)

print(json_data)
r = requests.post(url, json=json_data)
