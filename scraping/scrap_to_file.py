import pandas as pd
import time
from bs4 import BeautifulSoup
import requests

from add_by_scrap import scrap_by_users
import json

def scrap_to_f(users_urls):
    for url in users_urls:
        user_id = url.split('?')[-1].split('=')[-1]
        filename = 'data/'+user_id+'.json'
        revs = scrap_by_users(url)
        if revs is not None:
            with open(filename, 'w') as outfile:
                json.dump(revs, outfile)
