
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import numpy as np
import spacy
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')

def preprocessing(text):
    text = re.sub(r'\[\d+\]|\d+\]|[a-z]*\]|Q[0-9]*\)|Time :|\[Max.|\d{2}/\d{2}/\d{4}|Q.\d|OR', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def generateQues(text):
    questions = []
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cos_sim_matrix = cosine_similarity(vectors)
    threshold = 0.4
    pairs_above_threshold = np.where(cos_sim_matrix > threshold)
    pairs = [(i, j) for i, j in zip(pairs_above_threshold[0], pairs_above_threshold[1]) if i < j]
    groups = []
    for i, j in pairs:
        added = False
        for group in groups:
            if i in group or j in group:
                group.update([i, j])
                added = True
                break
        if not added:
            groups.append(set([i, j]))
    grouped_questions = []
    for group in groups:
        grouped_questions.append([sentences[sentence_idx] for sentence_idx in group])
    return grouped_questions

def findmatching(question_description,text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    vectorizer = TfidfVectorizer().fit_transform([question_description]+sentences)
    vectors = vectorizer.toarray()
    que_desc_vectors = vectors[0]
    pyq_vectors = vectors[1:]
    cos_sim_matrix = cosine_similarity([que_desc_vectors],pyq_vectors)
    threshold = 0.2
    pairs_above_threshold = np.where(cos_sim_matrix > threshold)
    pairs = [(i, j) for i, j in zip(pairs_above_threshold[0], pairs_above_threshold[1]) if i < j]
    groups = []
    for i, j in pairs:
        added = False
        for group in groups:
            if i in group or j in group:
                group.update([i, j])
                added = True
                break
        if not added:
            groups.append(set([i, j]))
    grouped_questions = []
    for group in groups:
        grouped_questions.append([sentences[sentence_idx] for sentence_idx in group])
    return grouped_questions

def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        text += page_text
    return text

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        text = ""
        if 'files[]' in request.files:
            files = request.files.getlist("files[]")
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                else:
                    text += file.read().decode('utf-8')

        questions = generateQues(text)
        cleaned_questions = []
        for group in questions:
            cleaned_group = [preprocessing(question) for question in group]
            cleaned_questions.append(cleaned_group)

        return render_template('questions.html', questions=cleaned_questions)

    return render_template('index.html')

@app.route("/matcher", methods=['POST'] )
def matcher():
    if request.method == 'POST':
        text=''
        question_description = request.form['question_description']
        if 'matchfiles' in request.files:
            files = request.files.getlist("matchfiles")
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                else:
                    text += file.read().decode('utf-8')

        matching_ques = findmatching(question_description,text)
        return render_template('index.html',matching_ques = matching_ques)
    
    return render_template('index.html.html')

if __name__ == "__main__":
    app.run(debug=True)
