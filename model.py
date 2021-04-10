import pickle
import pandas as pd

# contains one ML model and only one recommendation system that we have obtained from the
# previous steps to recommend top 5 products

def predict(username):
    '''
    Predicting the top recommended products using best ML models
    '''
    list_data = [[]]
    text_info = "Entered user name is not available. Please enter valid user name!"
    
    # load all input files
    review_df = pd.read_csv('./data/product_review.csv')
    sentiment_clean_df = pd.read_csv('./data/sentiment_df.csv')

    user_reco_file = open('./pickle/user_recommendation.pkl', 'rb')
    user_reco_table = pickle.load(user_reco_file)

    sentiment_model_file = open('./pickle/Sentiment_model.pkl', 'rb')
    sentiment_model = pickle.load(sentiment_model_file)

    tfidf_file = open('./pickle/tfidf_vectorizer.pkl', 'rb')
    tfidf_vector = pickle.load(tfidf_file)

    # check for valid username
    if username in user_reco_table.index:

        top20_product_ids = user_reco_table.loc[username].sort_values(ascending=False)[:20]
        product_map = pd.DataFrame(review_df[['id','name']]).drop_duplicates()
        top20_products = pd.merge(top20_product_ids, product_map, on='id')

        # Mapping product with product reviews
        product_mapping_review = pd.DataFrame(sentiment_clean_df[['id','text_data','user_sentiment']]).drop_duplicates()
        product_review_data =pd.merge(top20_products, product_mapping_review, on='id')

        # get features using tfidf vectorizer
        test_features= tfidf_vector.transform(product_review_data['text_data'])

        # Predict Sentiment Score on the above Product Reviews using the finally selected ML model
        product_review_data['predicted_sentiment'] = sentiment_model.predict(test_features)
        product_review_data['predicted_sentiment_score'] = product_review_data['predicted_sentiment'].replace(['negative','positive'],[0,1])

        # Find positive sentiment percentage for every product
        product_pivot = product_review_data.reset_index().pivot_table(values='predicted_sentiment_score', index='name', aggfunc='mean')
        product_pivot.sort_values(by='predicted_sentiment_score',inplace= True, ascending= False)
        
        # Get top 5 products
        list_data = [[index, out] for index, out in enumerate (product_pivot.head(5).index, 1)]
        text_info = "Top 5 Recommended products for \"" + username +  "\""

    return text_info, list_data
