# The-Right-Resume
Using ML techniques to improve resume based on job descriptions

# Project Motivations

Returning a higher yield on job application submittals


# Working Notes

## Day 1-3: 8/2 ~ 8/4
open source indeed scraper was only grabbing job previews. had to build a scraper that would extract the url's of te full job posting, and then navigate to each url and grab the contents of the website.
- resulted in issues

Attaining data has been a hurdle
- used apiscraper.com and got a free trial which gave thousands of url requests for free
- after all was said and done, attained a data
- indeed still had some sort of issues even with thhe proxy and i only could pull a couple hundred items at a time. 
- had classmates run my scraper for me after getting temporarily ip banned and my free proxy token stopped working as well. 

## Day 4: 8/5

Started finally getting a grasp of a way to tie in original project motivations.

Training the model to have distinguished profiles for each class/label (in this case, the job title), and utilize the profiles to provide a similarity matrix of the resume to each individual. As well as what words would better fit to 
- evaluating the model by testing it against its predictive power of classifying job titles after training it on job descriptions. 


early attempts at running a model.

Roadblocks:
- scraped some more data in the morning
- my dataset of 1977 entries was reduced to 1354
- problems with my term frequency detection
- * there seems to be an issue with some bigrams and unigrams feeding into the successive classes/labels...

```
# 'chemical+engineer':
  . Most correlated unigrams:
. project
. design
. process
. engineer
. engineering
  . Most correlated bigrams:
. related field
. genetic information
. fast pace
. full time
. problem solve
# 'data+scientist':
  . Most correlated unigrams:
. insight
. learn
. model
. science
. data
  . Most correlated bigrams:
. fast pace
. bachelor degree
. full time
. problem solve
. related field
# 'financial+analyst':
  . Most correlated unigrams:
. budget
. reporting
. accounting
. finance
. financial
  . Most correlated bigrams:
. full time
. fast pace
. communication skill
. related field
. bachelor degree
# 'physician':
  . Most correlated unigrams:
. license
. health
. physician
. patient
. care
  . Most correlated bigrams:
. fast pace
. communication skill
. related field
. bachelor degree
. full time
# 'recruiter':
  . Most correlated unigrams:
. interview
. source
. hire
. talent
. recruiting
  . Most correlated bigrams:
. full time
. problem solve
. related field
. job description
. fast pace
# 'underwriter':
  . Most correlated unigrams:
. policy
. relationship
. line
. risk
. insurance
  . Most correlated bigrams:
. fast pace
. full time
. related field
. communication skill
. bachelor degree
```


Multitude of classmates, DSR's and instructors support on this
- last but not least I wanna thank me



_____

Messy parts of data:

the '\n' was replaced when scraping in the HTML with an empty string. no spacing. this caused many words to merge together and the interpreter to read these unnaturally joined words as one. 


___
# Model Performance

## After Vectorizer Parameter Inputs

Pipeline(steps=[('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                ('clf',
                 RandomForestClassifier(max_depth=3, n_estimators=200,
                                        random_state=0))])
0.8643067846607669
                   precision    recall  f1-score   support

chemical+engineer       0.82      1.00      0.90        40
   data+scientist       0.83      1.00      0.90        38
financial+analyst       0.99      0.71      0.83       121
        physician       0.97      0.84      0.90        67
        recruiter       0.80      1.00      0.89        44
      underwriter       0.66      1.00      0.79        29

         accuracy                           0.86       339
        macro avg       0.84      0.92      0.87       339
     weighted avg       0.89      0.86      0.86       339

Pipeline(steps=[('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC())])
0.9793510324483776
                   precision    recall  f1-score   support

chemical+engineer       1.00      0.98      0.99        50
   data+scientist       1.00      0.94      0.97        49
financial+analyst       0.94      1.00      0.97        82
        physician       1.00      0.97      0.98        60
        recruiter       1.00      0.98      0.99        56
      underwriter       0.95      1.00      0.98        42

         accuracy                           0.98       339
        macro avg       0.98      0.98      0.98       339
     weighted avg       0.98      0.98      0.98       339

Pipeline(steps=[('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB())])
0.7315634218289085
                   precision    recall  f1-score   support

chemical+engineer       0.45      1.00      0.62        22
   data+scientist       0.37      1.00      0.54        17
financial+analyst       0.99      0.49      0.65       176
        physician       0.90      0.98      0.94        53
        recruiter       0.73      1.00      0.84        40
      underwriter       0.70      1.00      0.83        31

         accuracy                           0.73       339
        macro avg       0.69      0.91      0.74       339
     weighted avg       0.85      0.73      0.73       339

Pipeline(steps=[('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(random_state=0))])
0.9646017699115044
                   precision    recall  f1-score   support

chemical+engineer       1.00      1.00      1.00        49
   data+scientist       0.98      0.94      0.96        48
financial+analyst       0.94      0.95      0.95        86
        physician       0.97      0.93      0.95        60
        recruiter       0.98      0.98      0.98        55
      underwriter       0.93      1.00      0.96        41

         accuracy                           0.96       339
        macro avg       0.97      0.97      0.97       339
     weighted avg       0.97      0.96      0.96       339


## After Vectorizer Parameter Inputs

Pipeline(steps=[('vect',
                 CountVectorizer(max_df=0.75, max_features=5000, min_df=0.05,
                                 ngram_range=(1, 2),
                                 stop_words={'000', 'a', 'ability', 'about',
                                             'above', 'access', 'accomodation',
                                             'action', 'affirmative', 'after',
                                             'again', 'against', 'age', 'ain',
                                             'all', 'also', 'am', 'an', 'and',
                                             'any', 'applicant', 'are', 'area',
                                             'aren', "aren't", 'as', 'at', 'be',
                                             'because', 'been', ...})),
                ('tfidf', TfidfTransformer()),
                ('clf',
                 RandomForestClassifier(max_depth=3, n_estimators=200,
                                        random_state=0))])
0.943952802359882
                   precision    recall  f1-score   support

chemical+engineer       0.98      1.00      0.99        43
   data+scientist       0.91      1.00      0.95        40
financial+analyst       0.98      0.92      0.95        97
        physician       1.00      0.83      0.90        63
        recruiter       0.92      1.00      0.96        58
      underwriter       0.84      1.00      0.92        38

         accuracy                           0.94       339
        macro avg       0.94      0.96      0.94       339
     weighted avg       0.95      0.94      0.94       339

Pipeline(steps=[('vect',
                 CountVectorizer(max_df=0.75, max_features=5000, min_df=0.05,
                                 ngram_range=(1, 2),
                                 stop_words={'000', 'a', 'ability', 'about',
                                             'above', 'access', 'accomodation',
                                             'action', 'affirmative', 'after',
                                             'again', 'against', 'age', 'ain',
                                             'all', 'also', 'am', 'an', 'and',
                                             'any', 'applicant', 'are', 'area',
                                             'aren', "aren't", 'as', 'at', 'be',
                                             'because', 'been', ...})),
                ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
0.9616519174041298
                   precision    recall  f1-score   support

chemical+engineer       0.95      0.95      0.95        44
   data+scientist       0.95      0.93      0.94        45
financial+analyst       0.97      0.99      0.98        89
        physician       0.94      0.96      0.95        51
        recruiter       0.98      0.95      0.97        65
      underwriter       0.96      0.96      0.96        45

         accuracy                           0.96       339
        macro avg       0.96      0.96      0.96       339
     weighted avg       0.96      0.96      0.96       339

Pipeline(steps=[('vect',
                 CountVectorizer(max_df=0.75, max_features=5000, min_df=0.05,
                                 ngram_range=(1, 2),
                                 stop_words={'000', 'a', 'ability', 'about',
                                             'above', 'access', 'accomodation',
                                             'action', 'affirmative', 'after',
                                             'again', 'against', 'age', 'ain',
                                             'all', 'also', 'am', 'an', 'and',
                                             'any', 'applicant', 'are', 'area',
                                             'aren', "aren't", 'as', 'at', 'be',
                                             'because', 'been', ...})),
                ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
0.9587020648967551
                   precision    recall  f1-score   support

chemical+engineer       0.98      0.96      0.97        45
   data+scientist       0.93      0.98      0.95        42
financial+analyst       0.97      0.96      0.96        92
        physician       0.96      0.94      0.95        53
        recruiter       0.98      0.95      0.97        65
      underwriter       0.91      0.98      0.94        42

         accuracy                           0.96       339
        macro avg       0.96      0.96      0.96       339
     weighted avg       0.96      0.96      0.96       339

Pipeline(steps=[('vect',
                 CountVectorizer(max_df=0.75, max_features=5000, min_df=0.05,
                                 ngram_range=(1, 2),
                                 stop_words={'000', 'a', 'ability', 'about',
                                             'above', 'access', 'accomodation',
                                             'action', 'affirmative', 'after',
                                             'again', 'against', 'age', 'ain',
                                             'all', 'also', 'am', 'an', 'and',
                                             'any', 'applicant', 'are', 'area',
                                             'aren', "aren't", 'as', 'at', 'be',
                                             'because', 'been', ...})),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(random_state=0))])
0.9734513274336283
                   precision    recall  f1-score   support

chemical+engineer       0.98      0.98      0.98        44
   data+scientist       0.95      0.95      0.95        44
financial+analyst       0.98      0.99      0.98        90
        physician       0.98      0.96      0.97        53
        recruiter       0.98      0.95      0.97        65
      underwriter       0.96      1.00      0.98        43

         accuracy                           0.97       339
        macro avg       0.97      0.97      0.97       339
     weighted avg       0.97      0.97      0.97       339