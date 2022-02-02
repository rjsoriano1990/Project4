from flask import Flask, render_template,request,url_for
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('SPAM DATA.csv')
    cv = CountVectorizer()
    X=cv.fit_transform(df['Message'])
    y=df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=120)
    #ytb_model = open('YtbSpam_model.pkl','rb')
    #model = joblib.load(ytb_model)
    model = MultinomialNB()
    model.fit(X_train,y_train)
    model.score(X_test,y_test)
    
    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ =='__main__':
    app.run(debug=True)