import requests

resp = requests.post("http://127.0.0.1:1200/skin-cancer-diagnosis", files={'image': open(
    'new.png', 'rb')})

print(resp.text)

