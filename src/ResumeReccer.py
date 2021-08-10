import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

class ResumeReccer():

    def __init__(self,model) -> None:
        self.model = model
        self.keywords_dict = {}
        pass

    def predict_one(self,pipe, user_input):
        self.prediction = pipe.predict(pd.Series([user_input])).tolist()[0]
        return pipe.predict(pd.Series([user_input])).tolist()[0]

    def predict_one_proba(self,pipe, user_input):
        self.prediction_probabilities = pipe.predict_proba(pd.Series([user_input]))
        return pipe.predict_proba(pd.Series([user_input]))

    def get_keywords_dict(self,model):
        coefs = model.named_steps.clf.coef_
        feats = model.named_steps.vect.get_feature_names()

        # take coefficients and feature names, sorted? 

        dct = {"feats":feats,"coef":coefs[0,:].tolist()}

        titles = ["Chemical Engineer","Data Scientist","Financial Analyst","Physician","Recruiter","Underwriter"]

        d = dict()
        for i,title in enumerate(titles):
            coefs[i,:]
            dct = {"feats":feats,"coef":coefs[i,:].tolist()}
            top_words_df = pd.DataFrame(dct)
            top_words_df["coef"] = top_words_df["coef"].abs()
            top_words_df = top_words_df.sort_values("coef",ascending = False)
            a = top_words_df.feats.values.tolist()[:100]
            b = top_words_df.coef.values.tolist()[:100]
            d[title] = list(zip(a,b))

        self.keywords_dict = d

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

def clean_input(self,user_input):
    user_input = 
    
    self.user_input = user_input

cleaner 

predictors

