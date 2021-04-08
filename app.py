import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# prediction 
@app.route('/', methods=['POST', 'GET'])
def predict_recommendation():
    '''
    Predicting the top 5 recommendation product
    '''
    if request.method == 'POST':
        username = request.form['uname']
        out_data = [[]]

        # check for empty user name
        if len(username) == 0:
            infotext = "Please enter valid user name!!"
            return render_template('view.html', info=infotext, data=out_data, headings=['Index','Product'])
        
        # load all input files
        user_reco_file = open('./data/user_recommendation.pkl', 'rb')
        user_reco_matrix = pickle.load(user_reco_file)
        
        # check for valid username
        if username not in user_reco_matrix.index:
            infotext = "Entered user name is not present. Please enter valid user name!!"
            return render_template('view.html', info=infotext, data=out_data, headings=['Index','Product'])

        review_df = pd.read_csv('./data/product_review.csv')
        sentiment_df = pd.read_csv('./data/sentiment_df.csv')

        sentiment_model_file = open('./data/Sentiment_model.pkl', 'rb')
        sentiment_model = pickle.load(sentiment_model_file)

        tfidf_file = open('./data/tfidf_vectorizer.pkl', 'rb')
        tfidf_vector = pickle.load(tfidf_file)

        product_ids = user_reco_matrix.loc[username].sort_values(ascending=False)[:20]
        product_map = pd.DataFrame(review_df[['id','name']]).drop_duplicates()

        products = pd.merge(product_ids, product_map, on='id')
        # Mapping product with product reviews
        product_mapping_review = pd.DataFrame(sentiment_df[['id','text_data','user_sentiment']]).drop_duplicates()
        product_review_data =pd.merge(products, product_mapping_review, on='id')

        # get features using tfidf vectorizer
        tfidf_features = tfidf_vector.transform(product_review_data['text_data'])

        # Predict Sentiment Score on the above Product Reviews using the finally selected ML model
        product_review_data['predicted_sentiment'] = sentiment_model.predict(tfidf_features)
        product_review_data['predicted_sentiment_score'] = product_review_data['predicted_sentiment'].replace(['negative','positive'],[0,1])

        # Find positive sentiment percentage for every product
        product_pivot = product_review_data.reset_index().pivot_table(values='predicted_sentiment_score', index='name', aggfunc='mean')
        product_pivot.sort_values(by='predicted_sentiment_score',inplace= True, ascending= False)
        # Get top 5 products
        # product_map[product_map['name'] == out]['id']
        out_data = [[index, out] for index, out in enumerate (product_pivot.head(5).index, 1)]
        infotext = "Top Recommended products for \"" + username +  "\""
        return render_template('view.html', info=infotext, data=out_data, headings=['Index','Product'])  
    else:
        return render_template('view.html')  

# predicted_df = predict("00sab00")
# get the top products
# print(predicted_df.index)
# print(predicted_df.values)


if __name__ == '__main__':
    app.run(debug=True)