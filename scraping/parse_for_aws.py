import time
from bs4 import BeautifulSoup
import requests

import json

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
        count_li  = soup.select_one('li.review-count')
        if count_li is None:
            return None
        else:

            count_stro = count_li.select_one('strong')
            if count_stro is None:
                return None
            else:
                count_rev = int(count_stro.text)
                revs = []
                time.sleep(5)
                if count_rev > 0 :


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
                    time.sleep(5)
                return(revs)

    else:
        return None

def scrap_to_f(users_urls):
    for url in users_urls:
        user_id = url.split('?')[-1].split('=')[-1]
        filename = 'data/'+user_id+'.json'
        revs = scrap_by_users(url)
        if revs is not None:
            with open(filename, 'w') as outfile:
                json.dump(revs, outfile)

if __name__ == '__main__':
    file = open('th1_pr_urls.csv')
    ll = list(map(lambda x: x.strip(), file.readlines()))
    scrap_to_f(ll)
