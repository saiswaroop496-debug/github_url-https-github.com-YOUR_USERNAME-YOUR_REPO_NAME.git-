import re
import requests

url = "https://www.football-data.co.uk/data.php"
response = requests.get(url)
links = re.findall(r'href="([^"]+\.csv)"', response.text)

print("Found CSV links:")
for link in set(links):
    print(link)


print("Found CSV links:")
for link in links:
    print(link)
