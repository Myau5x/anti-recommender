import pandas as pd
from pymongo import MongoClient
import time
from yelp_api import *

mc = MongoClient()
db  = mc['raw_restaurants']
biz = db['restaurants']
users = db['users']
rv = db['reviews']



def build_restoraunts(zips):
    """ collects restaurant by zip code through YELP API"""
    for zip in zips:

        offsets = range(0,1000,50)

        for offset in offsets:
            response = search(API_KEY, DEFAULT_TERM, zip, offset)
            data = response.get('businesses', None)
            if data is None:
                break

            for row in data:
                add_to_database_if_new(row, biz)

def build_review_by_API():
    """Collects reviews by API, collects users from those reviews"""
    for business in biz.find():
        idx = business['id']

        response = get_reviews(API_KEY,idx)
        data = response.get('reviews')
        if data is not None:
            for row in response:
                add_to_database_if_new(row, rv)
                user = row['user']
                add_to_database_if_new(user, users)



def add_to_database_if_new(row, collection):
    """Add one row to DataBase"""
    if collection.count_documents({'id': row['id']}) == 0:
        collection.insert_one(row)

def retrieve_datatable(collection):
    """Return the contents of MongoDB collection as a dataframe."""
    return pd.DataFrame(list(collection.find({})))
