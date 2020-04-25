import urllib.request 
import json
import re


print("Started Reading JSON file")
with open("results12.json", "r") as read_file:
    items = json.load(read_file)

    print("Decoded JSON Data From File")
    for item in items:
       
        camId = re.search("\d{10}_(.*)\.jpg", item['url'])  
        print(item['url'] + " - " + camId.group())            
        urllib.request.urlretrieve(item['url'], camId.group())
        
        