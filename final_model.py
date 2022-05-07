import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from keras import models,layers

import warnings
warnings.filterwarnings('ignore')


def ml_model():

    df= pd.read_csv("ChronicKidneyDiseaseProcessed.csv")
    X=df.iloc[:,:-1]
    Y=df[["class"]]

    sc_X = StandardScaler()
    X= sc_X.fit_transform(X)


    pca = PCA(n_components = 17)
    X = pca.fit_transform(X)

    explained_ratio= pca.explained_variance_ratio_
    np.cumsum(explained_ratio)

    X=df.iloc[:,:12]
    x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8)

    model = models.Sequential()
    model.add(layers.Dense(512,activation='relu',input_dim=12))
    model.add(layers.Dense(216,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))
    #model.add(layers.Dense(84,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

    hist = model.fit(x_train,y_train,epochs = 200,verbose=0)
     
    neu_pr = np.round(model.predict(x_test))

    model.save('C:/Users/Karthika/flask_demo/SavedModels/DeepLearningModel')

    if neu_pr == 0:
        print("\n\n\n\t\t\t\t\t You are affected by Chronic disease")
    else:
        print("\n\n\n\t\t\t\t\t You are not affected")