import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class MyApp():

    def __init__(self):
        self.load()
    
    def load(self):
        self.model = pickle.load(open('../data/LogisticRegression.sav','rb'))

    def predict(self,resume):
        self.input_resume = resume
        self.occupation = self.model.predict(pd.Series([resume])).tolist()[0]
        return self.model.predict(pd.Series([resume])).tolist()[0]

    def get_keywords_dict(self):
        coefs = self.model.named_steps.clf.coef_
        feats = self.model.named_steps.vect.get_feature_names()

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

    def remove_stopwords(self,stopWords, descriptions):
        
        cleaned_descriptions = []
        for description in descriptions:
            temp_list = []
            for word in description.split():
                if word not in stopWords:
                    temp_list.append(word.lower())
            cleaned_descriptions.append(' '.join(temp_list))
        return cleaned_descriptions

    def remove_punctuation(self,descriptions):
        no_punct_descriptions = []
        for description in descriptions:
            description_no_punct = ' '.join(RegexpTokenizer(r'\w+').tokenize(description))
            no_punct_descriptions.append(description_no_punct)
        return no_punct_descriptions

    def get_wordnet_pos(self,word):
        # nltk.download()

        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_descriptions(self,descriptions):
        cleaned_descriptions = []
        for description in descriptions:
            temp_list = []
            for word in description.split():
                cleaned_word = WordNetLemmatizer().lemmatize(word, self,get_wordnet_pos(word))
                temp_list.append(cleaned_word)
            cleaned_descriptions.append(' '.join(temp_list))
        return cleaned_descriptions

    def clean_descriptions(self,topWords, descriptions):
        stopWords = set(stopwords.words('english'))
        no_punct = self.remove_punctuation(descriptions)
        no_punct_sw = self.remove_stopwords(stopWords, no_punct)
        cleaned = self.lemmatize_descriptions(no_punct_sw)
        return cleaned


    def clean_input(self,user_input):
        user_input = (user_input.split(" "))
        stopWords = set(stopwords.words('english'))
        self.cleaned_input = self.clean_descriptions(stopWords, user_input)
        return self.clean_descriptions(stopWords, user_input)

    
    def get_suggested_keywords(self):
    # Generate list of key words for that class
        self.get_keywords_dict()
        suggested_words,key_words = [],[]
        for i in range(len(self.keywords_dict[self.occupation])):
            key_words.append(self.keywords_dict[self.occupation][i][0])

    # add to suggested words list if resume doesn't contain matching words
        for i,key_word in enumerate(key_words):
            if i > 10 and suggested_words == []:
                contains_top_10 = True
                print("good job, you've got the top 10 words in your resume!")
                return []
            if key_word not in self.resume:
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

        print(suggested_words[:10])
        self.suggested_keywords = suggested_words[:10]
    
    def get_suggested_job_listings(self):
        cv = CountVectorizer()
        df = pd.read_pickle('../data/cleaned_job_descriptions.pickle')
        # index into df where it matches occupation

        desc_df = df[df['job_title']==self.occupation]

        desc_df=desc_df.drop(columns = ["Unnamed: 0", "Unnamed: 0.1"])

        descs = list(desc_df["job_desc"])

        d = {}
        for i in range(len(desc_df)):
            url = desc_df.iloc[i,1]
            desc = desc_df.iloc[i,2]
            desc_matrix = cv.fit_transform([self.input_resume,desc])
            d[url] = cosine_similarity(desc_matrix)[0][1]    

        sorted_d = [(k,v) for k, v in sorted(list(d.items()), key = lambda x : x[1])][::-1]
        print(sorted_d[:5])
        self.suggested_job_listings = sorted_d[:5]