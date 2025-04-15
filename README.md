# 🩺 Chronic Kidney Disease Prediction – Hands-on with AWS

An end-to-end machine learning project focused on predicting Chronic Kidney Disease (CKD) using patient data. This project encompasses data preprocessing, model development, and deployment, offering a practical application of machine learning in the healthcare domain.

# 📌 Project Overview

* This project aims to: Predict CKD based on patient health metrics. Implement a complete ML pipeline, from data collection to deployment.
  
# 🔍 Dataset

* **Source:** UCI Machine Learning Repository – Chronic Kidney Disease Dataset
* **Features:** 24 attributes including age, blood pressure, specific gravity, albumin, sugar, etc.
* **Target:** Classification into CKD or not.

# Project Structure
```
Chronic-Kidney-Disease-Prediction-hands-on-AWS/
├── data/
│   └── chronic_kidney_disease.csv
├── models/
│   └── trained_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
├── Chronic_disease_classification.ipynb
├── nextlabs.ipynb
└── README.md
```
# 📈 Methodology

* **Data Collection:** Acquired dataset from UCI repository.
* **Exploratory Data Analysis (EDA):** Visualized data distributions and relationships.
* **Data Preprocessing:** Handled missing values, Encoded categorical variables, Scaled numerical features
* Feature Selection: Identified significant features impacting CKD.

# Model Development:

* Implemented various classification algorithms.
* Evaluated models using accuracy, precision, recall, and F1-score.

# Model Deployment:

* Developed a Flask web application.
* Deployed the application using AWS services.

 # 🌐 Deployment on AWS

* The application is deployed using AWS services:
* EC2: Hosting the Flask application.
* S3: Storing static files and model artifacts.

# 📊 Results

* Achieved high accuracy in predicting CKD.
* The web application provides an intuitive interface for users to input patient data and receive predictions.

This is how my applications front page looks.
![front page](https://user-images.githubusercontent.com/93076299/172834454-ef1941b2-d635-48dc-8c96-4e721dfbe79b.PNG)
I provided two options. One is for knowing about the disease. 
And the second page is for prediction.
![page2](https://user-images.githubusercontent.com/93076299/172834715-05d378b6-8734-4d74-b148-f215f0e6474b.PNG)
![3rd](https://user-images.githubusercontent.com/93076299/172834754-d506310b-22a3-42e0-bdcc-fd2acbe52639.PNG)
And the prediction page looks like below
![4th](https://user-images.githubusercontent.com/93076299/172834851-f7243209-f802-4b96-b3ad-047d64b294c9.PNG)
