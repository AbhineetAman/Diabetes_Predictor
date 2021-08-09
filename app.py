# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 00:25:12 2021

@author: Abhineet
"""
from flask import Flask, render_template, request
import pickle
import numpy as np


with open('DCM.pkl', 'rb') as file1:
    my_model = pickle.load(file1)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict1():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = my_model.predict(data)

        if my_prediction == 1:
            return render_template('diabetes-positive.html')
        else:
            return render_template('diabetes-negative.html')


if __name__ == '__main__':
    app.run(debug=True)


