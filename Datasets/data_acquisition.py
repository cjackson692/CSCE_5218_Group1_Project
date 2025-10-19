import requests
import os
from io import BytesIO
url = 'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-tl.txt.zip'
file = 'en-tl.txt.zip'
path = '.' 

response = requests.get(url, stream=True)
response.raise_for_status()
with open(file, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"✅ Successfully downloaded '{file}'.")

