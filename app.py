import model
from flask import Flask, render_template, request

app = Flask(__name__)

# This is the Flask interface file to connect the backend ML models with the frontend HTML code
@app.route('/', methods=['POST', 'GET'])
def get_recommendations():
    '''
    Get top 5 recommended products using ML models
    '''
    if request.method == 'POST':
        username = request.form['uname']
        data_list = [[]]
        title=['Index', 'Product']
        text_info = "Invalid user! please enter valid user name."

        if len(username) > 0:
            text_info, data_list = model.predict(username)                
        return render_template('index.html', info=text_info, data=data_list, headings=title)  
    else:
        return render_template('index.html')  

if __name__ == '__main__':
    app.run(debug=True)