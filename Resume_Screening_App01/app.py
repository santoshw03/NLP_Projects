import sklearn
import spacy
import pickle
import streamlit as st

#load model:
model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

#cleanText function:
nlp = spacy.load("en_core_web_lg")
def cleanText(text):
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)


#Web App:
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Resume",type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #if utf-8 decoding fails try decoding with latin-1
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume =  cleanText(resume_text)

        input_features = tfidf.transform([cleaned_resume])

        prediction_id = model.predict(input_features)[0]
        
        #labels:
        labels = {
            'Data Science':0, 'HR':1, 'Advocate':2, 'Arts':3, 'Web Designing':4,
            'Mechanical Engineer':5, 'Sales':6, 'Health and fitness':7,
            'Civil Engineer':8, 'Java Developer':9, 'Business Analyst':10,
            'SAP Developer':11, 'Automation Testing':12, 'Electrical Engineering':13,
            'Operations Manager':14, 'Python Developer':15, 'DevOps Engineer':16,
            'Network Security Engineer':17, 'PMO':18, 'Database':19, 'Hadoop':20,
            'ETL Developer':21, 'DotNet Developer':22, 'Blockchain':23, 'Testing':24
        }

        # function to read values from labels:
        key_list = list(labels.keys())
        val_list = list(labels.values())
        def category_name(prediction_id):
            category_name = val_list.index(prediction_id)
            st.write("Classified Category: ",key_list[category_name])
        
        category_name(prediction_id)
        st.write(prediction_id)


#python main:
if __name__ == "__main__":
    main()    
