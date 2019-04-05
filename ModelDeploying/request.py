import requests
import json

url = 'http://localhost:5000/api'
with open('test.json') as json_file:
    json_data = json.load(json_file)

print(json_data)
r = requests.post(url, json=json_data)
