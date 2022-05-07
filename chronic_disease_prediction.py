import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

from keras import models,layers

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("ChronicKidneyDiseaseProcessed.csv")
X=df.iloc[:,:-1]
Y=df[["class"]]
#we are going to scale our data

sc_X = StandardScaler()
X= sc_X.fit_transform(X)

#Dimensionality reduction

from sklearn.decomposition import PCA
pca = PCA(n_components = 17)
X = pca.fit_transform(X)
explained_ratio= pca.explained_variance_ratio_
np.cumsum(explained_ratio)
X=df.iloc[:,:12]
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8)

# model building



## Support Vector Machine

svc_clf = SVC(kernel='linear') 
svc_clf.fit(x_train, y_train) 
svc_pr = svc_clf.predict(x_test)

filename = 'C:\Users\Karthika\flask_demo\SavedModels\svc.pkl'
pickle.dump(svc_clf, open(filename, 'wb'))

## K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn_pr=knn.predict(x_test)

filename = 'C:\Users\Karthika\flask_demo\SavedModels\knn_model.pkl'
pickle.dump(knn, open(filename, 'wb'))

## Decison Tree

dsn_tree = DecisionTreeClassifier(criterion="gini",splitter="random",random_state=10)
dsn_tree.fit(x_train,y_train)
dsn_pr = dsn_tree.predict(x_test)

filename = 'C:\Users\Karthika\flask_demo\SavedModels\decision_tree.pkl'
pickle.dump(dsn_tree, open(filename, 'wb'))


## Neural Network
model = models.Sequential()
model.add(layers.Dense(512,activation='relu',input_dim=12))
model.add(layers.Dense(216,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(84,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
hist = model.fit(x_train,y_train,epochs = 200,verbose=0)
neu_pr = np.round(model.predict(x_test))
model.save('C:/Users/Karthika/flask_demo/SavedModels/DeepLearningModel')