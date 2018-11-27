import pandas as pd
from pymongo import MongoClient
import time
from bs4 import BeautifulSoup
import requests
from starting_db import add_to_database_if_new

mc = MongoClient()
db  = mc['raw_restaurants']
biz = db['restaurants']
users = db['users']
rv = db['reviews']
rv_s = db['reviews_scrap']

### Must be used only by unscraped users
def parse_review(row):
    """parse one review"""
    rev = {}

    rev['id'] = row.get_attribute_list('data-review-id')[0]
    rev['rating'] = row.select_one('div.i-stars').get_attribute_list('title')[0].split()[0]
    rev['text'] = row.select_one('p').text
    if row.select_one('span.category-str-list') is not None:
        rev['category'] = row.select_one('span.category-str-list').text

    rev['date'] = row.select_one('span.rating-qualifier').text.strip()
    rev['alias'] = row.select_one('a.biz-name').attrs['href'].split('/')[-1]
    return rev

def scrap_by_users(user_url):
    """ user_url -- user's url. Return bizness reviews by those users"""
    user_id = user_url.split('?')[-1].split('=')[-1]
    add_start = 'https://www.yelp.com/user_details_reviews_self?rec_pagestart='
    response = requests.get(user_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        count_rev = int(soup.select_one('li.review-count').select_one('strong').text)
        revs = []
        time.sleep(1)
        if count_rev > 0:


            raw_reviews = soup.select('div.review')
    ### check that reviews > 0
            for row in raw_reviews:
                rev = parse_review(row)
                rev['user_id'] = user_id
                revs.append(rev)

        for page in range(10, count_rev, 10):
            url_add = add_start+str(page)+'&userid='+user_id
            response = requests.get(url_add)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                raw_reviews = soup.select('div.review')
                if raw_reviews is None:
                    break
                for row in raw_reviews:
                    rev = parse_review(row)
                    rev['user_id'] = user_id
                    revs.append(rev)
            time.sleep(1)
        return(revs)

    else:
        return None

def build_review_by_scrap(limit = 100):
    """collect all reviews for particular user"""
    count_unfind = 0
    for user in users.find({'rev_scrap': { '$exists' : False } }, limit = limit):
        user_url = user['profile_url']
        idx = user['id']
        data = scrap_by_users(user_url)

        if data is not None:
            users.update_one({'id': idx},{'$set': {'rev_scrap': '1'}})
            for row in data:
                ##accidentaly name it 'biz_id'
                row['biz_id'] = idx
                add_to_database_if_new(row, rv_s)

        else:
            count_unfind+=1
    print(count_unfind)
