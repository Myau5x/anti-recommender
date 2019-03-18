# anti-recommender
I often have to decide not where I really want to eat but filter places where I defenitelly don't want to go.
Yelp ratings don't help well because people care about different things and what is 5 for me is 2 for somebody, and vice versa.


## baseline

minimum result: create model that perform better than just predict all that worse 3.5 (or 4) is bad , all is more than 3.5 (4) is good. 

## data

For training models I used Yelp Academic Dataset available here : https://www.yelp.com/dataset

For validation I used data scraped from yelp. 
Folder `Scraping` contains code for web scraping and working with YELP API

First download data through yelp api for (king county zip codes)
files.

Then with BeautifulSoup I scraped user reviews for about 100 users 
files. Folder `data` contains examples of scraped data.

example how to do that in jupiter notebook `king_county_food.ipynb`

## Data preparation / modeling

### Finding hidden bad features using reviews

Filter reviews that is for restaurants

filter bad reviews

Split on test train

Using pyspark for this

Try ALS model for predicting rating but it predicts worse than mean rating. (jupiter notebooks and other sourse are in ALS folder)

Countvectorizing +  IDF  reviews

Using Kmeans for clustering

Than using clusters on review I assign cluster to restaurants, and to users on train set (every user/ restaurant can have several reviews)

If user dont like particular feature and restaurant have it I predict that it bad restaurant (User rate 1 or 2)

Check that for pair user/restaurants unseen in train test predicting bad rating works better

Save Kmeans cluster centroids, idf vector and countvectorising Vocabulary

Code for this: `nlp_model.py` and `NLP_tuning.ipynb` 

Save to csv basic restaurants info and predicted cluster `biz_cluster.csv`
Code for creating this file `save_biz.py`

### Predicting clusters using basic info about restauurants

Split `biz_cluster.csv` on train and test set

Drop features that Yelp dotsn't give through API
Create new feature rating/(number of reviews)

Train 16 Random forests and GradientBoosting Regressors for every clusters to predict if particular restaurant can be assigned to this cluster

Test it on test set

## Validation on scraped data

Create sklearn model working same as pyspark model using saved cluster centroids, idf vector and countvectorising Vocabulary
(https://github.com/Myau5x/anti-recommender/tree/master/model_parts)

using this model assign cluster to user based on their reviews

Assign clusters to restaurants using Random Forest (GradientBoostClassifier)

Predict if user rate restaurant as bad
Code for this in notebook `testing_on_scrap`

## Web site

On this moment web site works locally

 - User can give link to his profile on url 
 - My tool scrapes it
 - Clusters user according his bad reviews
 - Than user provide location
 - Tool calls Yelp API and takes first 100 restaurants for this location 
 - Predicts if those restaurants bad for user or not.

For easy using with Flask instead of trained pyspark model I created sklearn model working same way. Look on code here `rewrite_model_as_sklearn.ipynb` 
Web app works using Flask and Brython 
source code for this: `antirec.py` and `templates\index_2.html` Also `static\` need to Brython.

## Presentation
Presentation slides `Where not go to have lunch.pdf`










