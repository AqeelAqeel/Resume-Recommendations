import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv('./data/cleaned_job_descriptions')

df["job_id"] = df["job_title"].factorize()[0]
job_id_df = df[['job_id', 'job_title']].drop_duplicates().sort_values('job_id')
id_to_job = dict(job_id_df[['job_id', 'job_title']].values)
# Setting targets and training data
features = df['job_desc'].values
targets = df['job_id']

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


def _split_data(df):

    return train_test_split(cleaned_descriptions, df['job_title'].values)

def fit(model, X,y):
    pipe = Pipeline([('vect', CountVectorizer(stop_words=stopWords, min_df=0.1, max_df=0.75, max_features=500,ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('clf', model),
    ])
    pipe.fit(X,y)
    
    return pipe

def predict(pipe,X_test):
    return pipe.predict(X_test)

def predict_one(pipe, desc):
    return pipe.predict(pd.Series([desc])).tolist()[0]

def predict_one_proba(pipe, desc):
    return pipe.predict_proba(pd.Series([desc]))

if __name__ == '__main__':
  
    
    # Setting targets and training data
    descriptions = df['job_desc'].values
    targets = df['job_id']
    
    
    # Creating stop words
    stopWords = set(stopwords.words('english'))
    add_stopwords = {
        'join', 'work', 'team', 'future', 'digital', 'technology', 'access', 'leader', 'industry', 'history', 'innovation',
        'year', 'customer', 'focused', 'leading', 'business', 'ability', 'country', 'employee', 'www', 'seeking',
        'location', 'role', 'responsible', 'designing', 'code', 'ideal', 'candidate', 'also', 'duty', 'without', 'excellent',
        'set', 'area', 'well', 'use', 'strong', 'self', 'help', 'diverse', 'every', 'day', 'equal', 'employment', 'opportunity',
        'affirmative', 'action', 'employer', 'diversity', 'qualified', 'applicant', 'receive', 'consideration', 'regard',
        'race', 'color', 'religion', 'sex', 'national', 'origin', 'status', 'age', 'sexual', 'orientation', 'gender',
        'identity', 'disability', 'marital', 'family', 'medical', 'protected', 'veteran', 'reasonable', 'accomodation',
        'protect', 'status', 'equal', 'discriminate', 'hire', 'hiring','inclusive', 'diverse','benefits','vacation','000','10','nike',"trustpilot"
    }
    
    stopWords = stopWords.union(add_stopwords)

    # Initializing punctuation remover and lemmatizer
    tokenize_remove_punct = RegexpTokenizer(r'\w+')
    lemma = WordNetLemmatizer()

    # Cleaning descriptions for both the whole dataset and CO only
    cleaned_descriptions = clean_descriptions(stopWords, descriptions)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score,classification_report

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

X_train, X_test, y_train, y_test = _split_data(df)

for model in models:
    model = fit(model,X_train,y_train)
    predictions = predict(model,X_test)
    print(model)
    print(accuracy_score(predictions,y_test))
    print(classification_report(predictions,y_test))

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
    

resume = "AQEEL ALI Aqeelali0312@gmail.com | (408) 718-0712 | www.linkedin.com/in/aqeelali786 EDUCATION: California Polytechnic University, San Luis Obispo, CA B.S. Business Administration – Financial Management Concentration Minor in Psychology ● Honors: Principal’s List (3.5+ GPA for three consecutive academic terms) - Fall 2019 ● Relevant Coursework: Financial Engineering in Risk Management, Computer Applications in Finance, Advanced Corporate Finance Chartered Financial Analyst Level Two Candidate Exam Date: Nov 2021 ● Pursuing CFA designation by acquiring a wide breadth of portfolio management skills. ● Level One Exam passed on June 2019. WORK EXPERIENCE: Middle Market Portfolio Analyst – Comerica Bank, San Jose, CA Jul 2019 to Present ● Analyze employer’s Middle Market business for the California region through industry, financial, macroeconomic data and other supporting credit information concerning an applicant's credit requests. ● Identify key business and financial risks that may impact the repayment prospects by the borrower. ● Expertise in Salesforce Data Management and CRM software systems utilized while underwriting to 8-figure commercial banking facilities ranging from $5M to $100M loans and facilities. aggregate exposure of bank assets. ● Prepare, review and assess the creditworthiness of commercial loan originations and renewals by evaluating tax returns, spreads of financial statements, historical trends, rent rolls, leases, projections, management performance, industry reports, cash flow models, capital structure and collateral analysis and other relevant data to analyze portfolio companies’ repayment capacity. ● Ensure the integrity of performance data for clientele and prospects and maintain ongoing relationships with thecustodial partners banks, vendors, and internal groups. ● Offer insights into customer financial needs, including opportunities identified using Line of Business-approved relationship expansion tools. Contribute personal insights related to a loan structure's effectiveness to mitigate risks, appropriate to prevailing competitive market environment and Bank risk tolerances. ● Prepared & presented nation-wide internal quarterly Company Q1 & Q2 2020 earnings reports and portfolio updates (within a team of four). ● Undertook special project initiative while fluidly adapting self-starter work ethic to a remote work environment during the initial rollout of Federal Treasury Payroll Protection Program and reviewed numerous applicants’ eligibility & fund usage during COVID-19 global pandemic. Venture Capital Analyst Intern - LDR Ventures, San Luis Obispo, CA Jan 2019 to Apr 2019 ● Analyzed investment opportunities up to $1.5M, prepared fundraising pitches to external stakeholders and prospective investors, and identified potential risks for early stage portfolio companies ● Assisted in building pricing models to help companies launch multiple new product lines and conduct stress tests under varying scenario analyses. ● Oversaw a personally proposed initiative for a portfolio company’s marketing campaign across universities in California. LEADERSHIP & OTHER RELEVANT EXPERIENCE Banking Valuations, Investment Banking Society San Luis Obispo, CA Jan 2019 to Feb 2019 ● Took an extracurricular course which covered the three main methods of company valuations ● Competed in a Goldman Sachs case competition against over 20 teams to create a pitch deck and presentation for a real case study. Recommended a company’s IPO by analyzing their financial position, creating a pro forma financial model, computing value with several valuation methodologies and examining IPO market conditions Member - MacIntalkers of Toastmasters International Apple Cupertino, CA Jul 2018 to Feb 2020 ● Delivered five public speeches under the “Dynamic Leadership” Pathways project. ● Developed effective communication skills on a weekly basis. " 
lyft_job_desc = "Financial Analyst, Strategy Finance at Lyft San Francisco, CA At Lyft, our mission is to improve people’s lives with the world’s best transportation. To do this, we start with our own community by creating an open, inclusive, and diverse organization.  Lyft is hiring a Financial Analyst for its Strategy Finance Team. The candidate in this position will provide financial and analytical support to drive strategic decisions for the company and help prepare financial management reporting. As a Financial Analyst, you will work directly with stakeholders across Finance in forecasting, planning and reporting key metrics to senior leadership.  Responsibilities: Help in analyzing & modeling forecast trends for total company financials Assist in the preparation and analysis of consolidated P&L for actuals and forecasts, help the FP&A team on deliverables, ongoing variance analysis, and ad hoc modeling Help lead the FP&A team through weekly and monthly forecasting Assist in the quarterly and annual strategic planning process Collaborate with Investor Relations by analyzing relevant financial information in preparation for the earnings call and investor presentations Team up with Corporate Development to create Board of Directors financials Partner with FP&A, Accounting, Treasury, Tax, and HR to forecast centralized expenses Drive monthly and quarterly close activities for FP&A and support consolidated management reporting Partner with Accounting to manage close timelines, process and reporting Manage creation of internal executive reporting documents including board, close and other management presentations and workbooks Support initiatives to create process efficiencies & improvements within FP&A Experience: BA/BS with 3+ years of experience in financial planning and analytics (FP&A) in a rigorous environment Corporate Finance, forecasting, or consolidations experience is a plus Detail-oriented and organized self-starter with a drive to dig into complex problems Advanced Excel skills. Experience building complex formulas and manipulating large data sets Ability to work in a fast-paced, team-based environment with minimal supervision Research, quantitative and analytical skills Comfortable navigating through financial statements Ability to organize and track overlapping tasks and assignments, with frequent priority changes Strong interpersonal and communication skills, with the ability to communicate and influence effectively across various departments Benefits: Great medical, dental, and vision insurance options Mental health benefits In addition to 12 observed holidays, salaried team members have unlimited paid time off, hourly team members have 15 days paid time off 401(k) plan to help save for your future 18 weeks of paid parental leave. Biological, adoptive, and foster parents are all eligible Pre-tax commuter benefits Lyft Pink - Lyft team members get an exclusive opportunity to test new benefits of our Ridership Program " 
cmgr_job_desc = "About the Role: Raydiant is looking for a lover of technology, a customer-success driven channel manager to help build the channel and achieve set goals.  Customer Success Managers (Channel) coordinate and work closely with various channel partners and affiliates to build and grow the pipeline and sell our solutions to businesses. As a channel customer success manager, you will manage Raydiant’s business relationships within the US; working with customers from many different industries. You will help us achieve our mission of managing the channel partners we have, qualify and recruit more qualified partners to help drive business and grow the channel side of the business. This role is based in the office at our San Francisco, CA headquarters in the SOMA neighborhood. What You Will Be Doing:  Prospecting, qualifying and on-boarding channel partners that can help the company drive business growth month over month Be able to forecast monthly sales revenue and achieve sales goals and KPIs set by management hannel customer success managers will help manage all partners with a focus on creating yearly and quarterly channel plans Be able to travel and meet channel partners, to conduct presentations and live demos of the product Channel customer success managers will focus on training Raydiant partners on effective methods for selling, using the partner portal and using Raydiant’s product Channel customer success managers will build an excellent relationship with channel partners and have a focus on retaining sophisticated partners while coaching inexperienced Raydiant partners on the best sales practices. Channel customer success managers will create monthly sales reports and communicate channel partners monthly commission to them in coordination with the finance department To own management of the channel PRM To conduct webinars to attract new partners and conduct training sessions on new products and services   What We Are Looking For:   Passion for sales in B2B The motivation to go the extra mile with a positive can-do attitude Bachelor’s degree or equivalent experience Excellent communication and relationship-building relationship skills; you like to negotiate and to achieve targets  Strong software and new technology awareness At least 2-years experience in channel sales within a SaaS product line Proficient with Salesforce"  
ds_1_desc = "We are looking for a Data Scientist who will support our product, sales, leadership and marketing teams with insights gained from analyzing company data. The ideal candidate is adept at using large data sets to find opportunities for product and process optimization and using models to test the effectiveness of different courses of action. They must have strong experience using a variety of data mining/data analysis methods, using a variety of data tools, building and implementing models, using/creating algorithms and creating/running simulations. They must have a proven ability to drive business results with their data-based insights. They must be comfortable working with a wide range of stakeholders and functional teams. The right candidate will have a passion for discovering solutions hidden in large data sets and working with stakeholders to improve business outcomes.Responsibilities for Data ScientistWork with stakeholders throughout the organization to identify opportunities for leveraging company data to drive business solutions.Mine and analyze data from company databases to drive optimization and improvement of product development, marketing techniques and business strategies.Assess the effectiveness and accuracy of new data sources and data gathering techniques.Develop custom data models and algorithms to apply to data sets.Use predictive modeling to increase and optimize customer experiences, revenue generation, ad targeting and other business outcomes.Develop company A/B testing framework and test model quality.Coordinate with different functional teams to implement models and monitor outcomes.Develop processes and tools to monitor and analyze model performance and data accuracy.Qualifications for Data ScientistStrong problem solving skills with an emphasis on product development.Experience using statistical computer languages (R, Python, SLQ, etc.) to manipulate data and draw insights from large data sets.Experience working with and creating data architectures.Knowledge of a variety of machine learning techniques (clustering, decision tree learning, artificial neural networks, etc.) and their real-world advantages/drawbacks.Knowledge of advanced statistical techniques and concepts (regression, properties of distributions, statistical tests and proper usage, etc.) and experience with applications.Excellent written and verbal communication skills for coordinating across teams.A drive to learn and master new technologies and techniques.We’re looking for someone with 5-7 years of experience manipulating data sets and building statistical models, has a Master’s or PHD in Statistics, Mathematics, Computer Science or another quantitative field, and is familiar with the following software/tools:Coding knowledge and experience with several languages: C, C++, Java,JavaScript, etc.Knowledge and experience in statistical and data mining techniques: GLM/Regression, Random Forest, Boosting, Trees, text mining, social network analysis, etc.Experience querying databases and using statistical computer languages: R, Python, SLQ, etc.Experience using web services: Redshift, S3, Spark, DigitalOcean, etc.Experience creating and using advanced machine learning algorithms and statistics: regression, simulation, scenario analysis, modeling, clustering, decision trees, neural networks, etc.Experience analyzing data from 3rd party providers: Google Analytics, Site Catalyst, Coremetrics, Adwords, Crimson Hexagon, Facebook Insights, etc."

resume = "AQEEL ALI Aqeelali0312@gmail.com | (408) 718-0712 | www.linkedin.com/in/aqeelali786 EDUCATION: California Polytechnic University, San Luis Obispo, CA B.S. Business Administration – Financial Management Concentration Minor in Psychology ● Honors: Principal’s List (3.5+ GPA for three consecutive academic terms) - Fall 2019 ● Relevant Coursework: Financial Engineering in Risk Management, Computer Applications in Finance, Advanced Corporate Finance Chartered Financial Analyst Level Two Candidate Exam Date: Nov 2021 ● Pursuing CFA designation by acquiring a wide breadth of portfolio management skills. ● Level One Exam passed on June 2019. WORK EXPERIENCE: Middle Market Portfolio Analyst – Comerica Bank, San Jose, CA Jul 2019 to Present ● Analyze employer’s Middle Market business for the California region through industry, financial, macroeconomic data and other supporting credit information concerning an applicant's credit requests. ● Identify key business and financial risks that may impact the repayment prospects by the borrower. ● Expertise in Salesforce Data Management and CRM software systems utilized while underwriting to 8-figure commercial banking facilities ranging from $5M to $100M loans and facilities. aggregate exposure of bank assets. ● Prepare, review and assess the creditworthiness of commercial loan originations and renewals by evaluating tax returns, spreads of financial statements, historical trends, rent rolls, leases, projections, management performance, industry reports, cash flow models, capital structure and collateral analysis and other relevant data to analyze portfolio companies’ repayment capacity. ● Ensure the integrity of performance data for clientele and prospects and maintain ongoing relationships with thecustodial partners banks, vendors, and internal groups. ● Offer insights into customer financial needs, including opportunities identified using Line of Business-approved relationship expansion tools. Contribute personal insights related to a loan structure's effectiveness to mitigate risks, appropriate to prevailing competitive market environment and Bank risk tolerances. ● Prepared & presented nation-wide internal quarterly Company Q1 & Q2 2020 earnings reports and portfolio updates (within a team of four). ● Undertook special project initiative while fluidly adapting self-starter work ethic to a remote work environment during the initial rollout of Federal Treasury Payroll Protection Program and reviewed numerous applicants’ eligibility & fund usage during COVID-19 global pandemic. Venture Capital Analyst Intern - LDR Ventures, San Luis Obispo, CA Jan 2019 to Apr 2019 ● Analyzed investment opportunities up to $1.5M, prepared fundraising pitches to external stakeholders and prospective investors, and identified potential risks for early stage portfolio companies ● Assisted in building pricing models to help companies launch multiple new product lines and conduct stress tests under varying scenario analyses. ● Oversaw a personally proposed initiative for a portfolio company’s marketing campaign across universities in California. LEADERSHIP & OTHER RELEVANT EXPERIENCE Banking Valuations, Investment Banking Society San Luis Obispo, CA Jan 2019 to Feb 2019 ● Took an extracurricular course which covered the three main methods of company valuations ● Competed in a Goldman Sachs case competition against over 20 teams to create a pitch deck and presentation for a real case study. Recommended a company’s IPO by analyzing their financial position, creating a pro forma financial model, computing value with several valuation methodologies and examining IPO market conditions Member - MacIntalkers of Toastmasters International Apple Cupertino, CA Jul 2018 to Feb 2020 ● Delivered five public speeches under the “Dynamic Leadership” Pathways project. ● Developed effective communication skills on a weekly basis. " 

def clean_input(resume):
    resume = (resume.split(" "))
    stopWords = set(stopwords.words('english'))
    return clean_descriptions(stopWords, resume)

ce_resume = "Tim Kasteler, Chemical Engineer tim.q.kasteler@gmail.com linkedin.com/in/timqkasteler641-234-1466 Professional Summary Perceptive chemical engineer with 2+ years of experience. Skilled in process design and project management. Seeking to deliver out-of-the-box solutions at Agaffre, inc. At Lesiliti, lowered equipment malfunctions by 20% through improved work procedures and maintenance. Raised throughput 25% by designing two new production processes. Work Experience Chemical EngineerLesiliti, Inc.Feb 2017–May 2019    Slashed equipment malfunctions by 20% with improved work procedures and maintenance.    Trained 20 technicians and chemists in production best practices, cutting defects by 15%.    Designed and implemented new changeover procedures that saved 18 labor hours per week.    Increased throughput 25% through design of two new production processes. ChemistTrukgill, Inc.Feb 2016–Jan 2017    Developed new waste-stream treatment process that reduced waste output by 18%.    Created a new technique to retrieve by-products that saved $20,000 a year. Education 2011–2015 University of Northern IowaBachelor of Science in Chemical Engineering    Pursued a passion for process design coursework.    Conducted project in waste stream management that was written up in IChemE blog. Skills     Technical Skills: Project management, process design, testing, management    Soft Skills: Interpersonal skills, collaboration, communication, efficiency Activities   Leader of weekly fishing club.  financial  Article, “Waste Stream Management” published in Chemical Processing Blog."
occupation = predict_one(model, ce_resume)
# ce_resume = clean_input(ce_resume)

# get difference between words 
def suggested_keywords(resume,occupation):
# Generate list of key words for that class
    suggested_words,key_words = [],[]
    for i in range(len(d[occupation])):
        key_words.append(d[occupation][i][0])

# add to suggested words list if resume doesn't contain matching words
    for i,key_word in enumerate(key_words):
        if i > 10 and suggested_words == []:
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

    return suggested_words[:10]


if suggested_keywords(clean_input(resume),occupation) == []:
    print("good job, you've got the top 10 words in your resume!")
    
else: 
    print(suggested_keywords(clean_input(resume),occupation))


