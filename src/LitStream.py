import streamlit as st

from NewPy import MyApp


import streamlit as st
import numpy as np
import joblib

#Interface
st.markdown('## Resume Reccomendation')

user_input = st.text_area(label = "Copy your Resume text",value='Paste your resume text here', height=30, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None)

user_input = user_input.replace('\r', ' ').replace('\n', ' ')

#Predict button
if st.button('Improve my Resume!'):
    model= MyApp() 
    model.predict(user_input)
    
    # model.suggested_keywords(), model.get_suggested_job_listings()


    # render_template('resume_reccomendations.html', pred= 'Your resume is classified as a {}. Here are my improvements to your resume for this occupation: {}. The top 5 jobs you are currently most qualified for in terms of similiary are: {}'.format(model.predict(user_input),model.suggested_keywords(), model.get_suggested_job_listings()))
    job_links = []
    for i in range(len(model.get_suggested_keywords())):
        job_links.append(model.get_suggested_keywords()[i])

    for i in range(len(job_links)):
        st.markdown(f'### Recommended word #{i+1}: {job_links[i]}')

    # job_links = [x for x in model.get_suggested_job_listings()[:5]]
    job_links = []
    for i in range(len(model.get_suggested_job_listings())):
        job_links.append(model.get_suggested_job_listings()[i])

    for i in range(len(job_links)):
        st.markdown(f'### [Job #{i+1}]({job_links[i][0]})')
