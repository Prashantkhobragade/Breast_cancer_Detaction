from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

#Route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicted', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        pass