import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
url="https://store.steampowered.com/appreviews/"
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
    cursor = "*"
    params = {
        "json" :1,
        "key": "79A88DB2911C0F3B4D6835EFA8306C83",
        "appid": app_id,
        "filter": "recent",
        "review_type" : "all",
        "language": "english"
    }
    reviews = []
    n = 50000 # set the number of reviews you want to fetch

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100,n)
        response = requests.get(url=url+app_id, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception if the response is not OK
        reviews_json= json.loads(response.content)
        reviews += reviews_json["reviews"]
        if "cursor" in reviews_json:
            cursor = reviews_json["cursor"]
        else:
            break
        n -= 100
    
    return reviews

# game_name = "Grand Theft Auto V"
# app_id = get_app_id(game_name)
# reviews = get_reviews(app_id)

# df = pd.DataFrame(reviews)
# df.to_csv('GTA_V.csv', index =False)

game_2_name= 'Grand Theft Auto Chinatown Wars' # This title belongs to bundle of GTA IV: The Complete Edition
app_id_2 = get_app_id(game_2_name)


reviews_2 = get_reviews(app_id_2)
df_2 = pd.DataFrame(reviews_2)
df_2.info()
df_2.to_csv('GTA_IV.csv', index=False)

game_3_name = 'Grand Theft Auto: San Andreas'
app_id_3 = get_app_id(game_3_name)
app_id_3