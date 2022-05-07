import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def preproc():
    df=pd.read_csv("chronic_kidney_disease_full.csv",na_values= "?")
    df=df.drop(["pc","pcc","ba","htn","dm","cad","pe"],axis=1) 
    num_col = df.select_dtypes(include='float64')
    for col in num_col:
        df[col].fillna(np.round(df[col].median(),2),inplace=True)
    df['appet']=df['appet'].fillna(df['appet'].mode()[0])
    df['ane']=df['ane'].fillna(df['ane'].mode()[0])
    encoder = LabelEncoder()
    df['class']= encoder.fit_transform(df['class'])

    df['appet']= encoder.fit_transform(df['appet'])

    df['ane']= encoder.fit_transform(df['ane'])
    train_df = df.dropna() ## data frame without NaN values
    test_df = df[df['rbc'].isna()]
    encoder = LabelEncoder()
    train_df['rbc']= encoder.fit_transform(train_df['rbc'])

    rbc=train_df['rbc']
    rbc_train_df=train_df.drop("rbc",axis=1)
    rbc_train_df['rbc']=rbc

    rbc=test_df['rbc']
    rbc_test_df=test_df.drop("rbc",axis=1)
    rbc_test_df['rbc']=rbc

    rbc_x_train = rbc_train_df.iloc[:,:-1]
    rbc_y_train =rbc_train_df[["rbc"]]
    rbc_x_test = rbc_test_df.iloc[:,:-1]
    rbc_y_test = rbc_test_df[["rbc"]]


    model = LogisticRegression()
    model.fit(rbc_x_train,rbc_y_train)
    pred = model.predict(rbc_x_test)
    test_df['rbc']=pred

    new_df=pd.concat([train_df,test_df],ignore_index=True) #concatinatiing test and train data previously splited

    #saving as csv
    #new_df.to_csv("ChronicKidneyDiseaseProcessed.csv")

