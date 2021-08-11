from flask import Flask,render_template,request

app = Flask(__name__)

from app import routes

app.run(host='0.0.0.0', port=8887, debug=True)