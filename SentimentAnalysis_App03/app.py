from flask import Flask , request , render_template
import sklearn
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle


model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('cv.pkl','rb'))

app = Flask(__name__)

#text cleaning:
stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(D|P)')

def preprocessing(text):
    text = re.sub('<[^>]*>','',text)
    emojis = emoji_pattern.findall(text)
    text = re.sub('[\W+]',' ',text.lower()) + ' '.join(emojis).replace('-', '')
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analysis', methods = ['POST'])
def analysis():
    if request.method == 'POST':
        comment = request.form['comment']
        cleaned_comment = preprocessing(comment)
        comment_vector = cv.transform([cleaned_comment])
        prediction =  model.predict(comment_vector)[0]

        return render_template('index.html',prediction = prediction , comment = comment)

        
if __name__ == '__main__':
    app.run(debug=True)