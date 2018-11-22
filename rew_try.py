from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib


# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib2 import HTTPError
    from urllib import quote
    from urllib import urlencode

import pandas as pd
from sample import request


# Yelp Fusion no longer uses OAuth as of December 7, 2017.
# You no longer need to provide Client ID to fetch Data
# It now uses private keys to authenticate requests (API Key)
# You can find it on
# https://www.yelp.com/developers/v3/manage_app
API_KEY= open('api.txt').read().strip()


# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
REVIEW_PATH = '/businesses/{id}/reviews'

# Defaults for our simple example.
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 50
SEARCH_OFFSET =0

def get_reviews(api_key, business_id):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id +'/reviews'

    return request(API_HOST, business_path, api_key)

def main():
    b50 = pd.read_csv('sea_50.csv')

    b50['id']= b50['id'].str.strip()

    rev50 = open('rev50.json','w')
    u50 = {}
    for idx in b50['id']:
        try:
            response = get_reviews(API_KEY,idx)
            rev50.write(str(response))
            rev3 = response.get('reviews')

            if not rev3:
                continue

            for re in rev3:
                u50[re['user']['id']] = re['user']['profile_url']



        except HTTPError as error:
            sys.exit(
                'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                    error.code,
                    error.url,
                    error.read(),
                )
            )
    user50 = pd.Series(u50)
    user50.to_csv('user50.csv')


if __name__ == '__main__':
    main()
