import base64
import requests

url = 'http://127.0.0.1:8000/upload'
with open("1.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    
payload ={"filename": "photo.png", "filedata": encoded_string}
resp = requests.post(url=url, data=payload) 