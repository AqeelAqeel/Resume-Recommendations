import streamlit as st


from NewPy import MyApp

st.write("some shit")

import streamlit as st
import numpy as np
import joblib

#Interface
st.markdown('## Resume Reccomendation')
user_input = st.text_input('Copy and Paste your resume here')

#Predict button
if st.button('Improve my Resume!'):
    model= MyApp() 
    model.predict(user_input)
    
    # model.suggested_keywords(), model.get_suggested_job_listings()
    print(user_input)

    # render_template('resume_reccomendations.html', pred= 'Your resume is classified as a {}. Here are my improvements to your resume for this occupation: {}. The top 5 jobs you are currently most qualified for in terms of similiary are: {}'.format(model.predict(user_input),model.suggested_keywords(), model.get_suggested_job_listings()))
        

    st.markdown('### Inputs must be a single line block of text')
    
    st.markdown(f'### Prediction is {model.get_suggested_keywords()}')

    st.markdown(f'### Top URLs are {model.get_suggested_job_listings()}')

# model= MyApp()

# @app.route('/predict',methods=['POST'])
# def predict():  
    
#     user_input = request.get_json()
#     print(user_input)
#     return render_template('resume_reccomendations.html', pred= 'Your resume is classified as a {}. Here are my improvements to your resume for this occupation: {}. The top 5 jobs you are currently most qualified for in terms of similiary are: {}'.format(model.predict(user_input),model.suggested_keywords(), model.get_suggested_job_listings()))
        
# if  __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8887, debug=True)
#     # app.run(host='0.0.0.0', port=8887, debug=True,ssl_context='adhoc')

from flask import Flask,request, url_for, redirect, render_template
from NewPy import MyApp

app = Flask(__name__)

model= MyApp()

@app.route('/predict',methods=['POST'])
def predict():  
    
    user_input = request.get_json()
    print(user_input)
    return render_template('resume_reccomendations.html', pred= 'Your resume is classified as a {}. Here are my improvements to your resume for this occupation: {}. The top 5 jobs you are currently most qualified for in terms of similiary are: {}'.format(model.predict(user_input),model.suggested_keywords(), model.get_suggested_job_listings()))
        
if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=8887, debug=True)
    # app.run(host='0.0.0.0', port=8887, debug=True,ssl_context='adhoc')



