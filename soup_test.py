from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib

import pandas as pd

from bs4 import BeautifulSoup

def scrape_rev(url, sel):
    """Return a list of lists (rows) of data from a table.
  look at NBA testik ~/galvanize
    Arguments
    ---------
    url : str
        The URL of the site to scrape.
    sel : str
        The CSS selector of the table to scrape.
    """
    responce = requests.get(url)
    html = responce.content
    soup = BeautifulSoup(html,'html.parser')
    time.sleep(10)
    table = soup.select_one(sel)
    table_rows = table.select('tr')
    sel = 'div.review'
    ten_rev =  soup.select(sel)

    new_rest = []
    for r  in ten_rev:
        #### filter by categories
        new_rest.append(r.select_one('a.js-analytics-click'))

    return new_rest


def main():
    u50 = pd.read_csv('user50.csv')
    for url in u50['url']:
        scrape_table_soup(url, 'div.review')
        ###Check how many reviews , we have to do n//10 scraping,
        ### url = 'https://www.yelp.com/user_details?userid=uVOLLjTnrDWHQ7qIjlLMvA'
        ### adding url_next1 = 'https://www.yelp.com/user_details_reviews_self?rec_pagestart=10&userid=b9z-0ug0n8lyRoWYsxQ_kQ'




if __name__ == '__main__':
    main()
