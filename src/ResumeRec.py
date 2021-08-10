from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.pipeline import Pipeline

app = Flask(__name__)

model=pickle.load(open('../data/LogisticRegression.sav','rb'))


def predict(pipe,X_test):
    return pipe.predict(X_test)

def predict_one(pipe, desc):
    return pipe.predict(pd.Series([desc])).tolist()[0]

def predict_one_proba(pipe, desc):
    return pipe.predict_proba(pd.Series([desc]))

def remove_stopwords(stopWords, descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            if word not in stopWords:
                temp_list.append(word.lower())
        cleaned_descriptions.append(' '.join(temp_list))
    return np.array(cleaned_descriptions)

def remove_punctuation(descriptions):
    no_punct_descriptions = []
    for description in descriptions:
        description_no_punct = ' '.join(RegexpTokenizer(r'\w+').tokenize(description))
        no_punct_descriptions.append(description_no_punct)
    return np.array(no_punct_descriptions)

def get_wordnet_pos(word):
    # nltk.download()

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
               'N': wordnet.NOUN,
               'V': wordnet.VERB,
               'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_descriptions(descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            cleaned_word = WordNetLemmatizer().lemmatize(word, get_wordnet_pos(word))
            temp_list.append(cleaned_word)
        cleaned_descriptions.append(' '.join(temp_list))
    return np.array(cleaned_descriptions)

def clean_descriptions(stopWords, descriptions):
    no_punct = remove_punctuation(descriptions)
    no_punct_sw = remove_stopwords(stopWords, no_punct)
    cleaned = lemmatize_descriptions(no_punct_sw)
    return cleaned

def get_representative_jobs(df, kmeans):
    cluster_centers = kmeans.cluster_centers_
    for cent in cluster_centers:
        print('\nCluster Represnetations')
        dist = euclidean_distances(cent.reshape(1,-1), tfidf)
        order = np.argsort(dist)
        for o in order[0][:5]:
            title = df['job_title'].iloc[o]
            print(title)

def clean_input(resume):
    resume = (resume.split(" "))
    stopWords = set(stopwords.words('english'))
    tokenize_remove_punct = RegexpTokenizer(r'\w+')
    lemma = WordNetLemmatizer()

    # Cleaning descriptions for both the whole dataset and CO only
    return clean_descriptions(stopWords, resume)

def suggested_keywords(resume,occupation):
# Generate list of key words for that class
    suggested_words,key_words = [],[]
    for i in range(len(d[occupation])):
        key_words.append(d[occupation][i][0])

# add to suggested words list if resume doesn't contain matching words
    for i,key_word in enumerate(key_words):
        if i > 15 and suggested_words == []:
            contains_top_10 = True
            return []
        if key_word not in resume:
            suggested_words.append(key_word)

# Look through list of needed words
# If any word stems begin with list of words to change, change it to a understandable output.
    start_of_words_to_change = ["financ","analy","engineer", "model",]
    for i,word in enumerate(suggested_words):
        for check in start_of_words_to_change:
            if word.startswith(check):
                if check == "financ":
                    suggested_words[i] = "Variations of the word: finance"

                elif check == "analy":
                    suggested_words[i] = "Variations of the word: analysis"

                else:
                    suggested_words[i] = f"Variations of the word: {check}"

    return suggested_words[:5]

@app.route('/')
def hello_world():
    
    return render_template("resume_reccomendations.html")

#Take in the resume and return my outputs
# Forms . request to get resume  !? 

@app.route('/predict',methods=['POST','GET'])
def predict():
    occupation = predict_one(model,clean_input(ce_resume))
    output='{0:.{1}f}'.format(prediction[0][1], 2)
  
    return render_template('resume_reccomendations.html', pred= 'Your resume is classified as a {}. Here are my improvements to your resume for this occupation: {}'.format(occupation,suggested_keywords(clean_input(ce_resume),occupation)))
        
        
if __name__ == '__main__':
    app.run(debug=True)
    
