import requests
import pandas as pd
from bs4 import BeautifulSoup
import json

def get_app_id(game_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {
        "key": "79A88DB2911C0F3B4D6835EFA8306C83",
        "term": game_name,
        "category1": 998
    }
    response = requests.get("https://store.steampowered.com/search/", headers=headers, params=params)
    response.raise_for_status()  # Raise an exception if the response is not OK
    soup = BeautifulSoup(response.text, 'html.parser')
    app_id = soup.find(class_='search_result_row')['data-ds-appid']
    return app_id

def get_reviews(app_id):
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {
        "key": "79A88DB2911C0F3B4D6835EFA8306C83",
        "appid": app_id,
        "filter": "recent",
        "language": "english"
    }
    response = requests.get("https://api.steampowered.com/appreviews/", headers=headers, params=params)
    response.raise_for_status()  # Raise an exception if the response is not OK
    reviews_json= json.loads(response.content)
    reviews = reviews_json["reviews"]
    return reviews

game_name = "Grand Theft Auto V"
app_id = get_app_id(game_name)
reviews = get_reviews(app_id)



