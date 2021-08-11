from flask import Flask, app,render_template,request

from NewPy import MyApp

model= MyApp()
app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

@app.route('/predict',methods=['POST'])
def predict():  
    
    user_input = request.get_json()
    print(user_input)
    return render_template('resume_reccomendations.html', pred= 'Your resume is classified as a {}. Here are my improvements to your resume for this occupation: {}. The top 5 jobs you are currently most qualified for in terms of similiary are: {}'.format(model.predict(user_input),model.suggested_keywords(), model.get_suggested_job_listings()))
        
if  __name__ == '__main__':
    model.run(host='0.0.0.0', port=8887, debug=True)
    # app.run(host='0.0.0.0', port=8887, debug=True,ssl_context='adhoc')


