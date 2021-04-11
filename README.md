# Capstone_RecoSystem
This is an upGgrad assigned capstone project [Sentiment-based product recommendation system]

Explanation for each file present current directory.

jupyter:
./jupyter/Sentiment Product Recommendation Capstone.ipynb:
 An end-to-end Jupyter      Notebook, which consists of the entire code (data cleaning and processing steps, text processing steps, feature extraction, four ML models, two recommendation systems and their evaluations, recommendation code and fine-tuning code of the recommendation system, etc.)

data:
./data/product_review.csv --> input raw data [product data]

./data/sentiment_df.csv --> clean data frame to get the reviews of predicted users from user recommedation model

pickle:
./pickle/user_recommendation.pkl --> user - user recommnedation model

./pickle/tfidf_vectorizer.pkl --> tfidf vectorizer object to convert the text to features

./pickle/Sentiment_model.pkl --> sentiment analysis best model [logsistic regression]

static:
./static/style.css --> style sheet for index.html

templates:
./templates/index.html --> web page view is designed in this html file

app.py --> interface for flask api to connect ML models with webpage

model.py --> ML related computional functions and reading all the input files

procfile & requirements.txt --> heroku deployment related files

This project is deployed under heroku website in the below link:
https://dashboard.heroku.com/apps/recommendation-system-capstone
