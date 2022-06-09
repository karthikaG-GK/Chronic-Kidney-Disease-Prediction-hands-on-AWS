from flask import Flask, render_template, request
import numpy as np
import pickle


model = pickle.load(open('models/logistic.pkl', 'rb'))

app = Flask(__name__)



@app.route("/")
def home():

    title = 'Kidney Disease Identify'
    return render_template("index.html", title = title)

@app.route("/diseaseinfo")
def disease():
    return render_template("diseaseinfo.html")

@app.route("/details")
def info():
    return render_template("details.html")

@app.route("/disease_prediction", methods =['POST'])
def disease_prediction():

    title = 'Chronic Kidney Disease Prediction'
    info1 = request.form['age']
    info2 = request.form['blood_preasure']
    info3 = request.form['Specific_Gravity']
    info4 = request.form['Albumin']
    info5 = request.form['Sugar']
    info6 = request.form['Red_Blood_cells']
    if info6=="normal":
        info6 = 0
    else:
        info6 = 1
    info7 = request.form['Blood_Glucose']
    info8 = request.form['Blood_Urea']
    info9 = request.form['Serum_Creatinine']
    info10 = request.form['Sodium']
    info11 = request.form['Potassium']
    info12 = request.form['Hemoglobin']  

    x_value = np.array([[info1, info2, info3, info4, info5, info6, info7, info8, info9, info10, info11, info12]])
    predicted = model.predict(x_value)
           
    return render_template("disease_prediction.html", data = predicted, title = title)


if __name__ == "_main_":
    app.run(host='0.0.0.0', port=8080)