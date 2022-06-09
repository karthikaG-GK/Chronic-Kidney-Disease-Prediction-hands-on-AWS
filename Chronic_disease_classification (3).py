#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import sklearn
sklearn.__version__


# # Exploring the data

# ### Chronic Kidney Disease Dataset upload

# In[3]:


df=pd.read_csv("chronic_kidney_disease_full.csv",na_values= "?")


# In[4]:


df.head(20)


# In[9]:


#df.info()


# In[10]:


#df.isna().sum()


# In[11]:


sns.heatmap(df.corr(),annot=True)


# # Data Preprocessing
# 
# ### Steps:
# ### 1. Drop unwanted features/variables/attributes
# ### 2. Replace missing values in numerical columns
# ### 3. Categorical Data encoding
# ### 4. Repalce missing values in categorical data
# ### 5. Split train and test data
# ### 6. Feature Scaling

# #### Droping Unecessary attributes

# In[5]:


df=df.drop(["pc","pcc","ba","htn","dm","cad","pe"],axis=1) 


# In[6]:


sns.heatmap(df.corr(),annot=True)


# #####  Handling Missing Values:
# 
# #####            Going to replace missing values in numerical column with their corresponding means.

# In[7]:


num_col = df.select_dtypes(include='float64')
for col in num_col:
    df[col].fillna(np.round(df[col].median(),2),inplace=True)
# now all the numerical columns are replaced with median of that corresponding columns
# now there is no numerical column with missing values


# In[8]:


"""
   only we have 3 categorical columns. On that columns, appet and ane have only 1 missing value. 
We just replace NaN with most frequent values. Now we left with only one categorical column. 
Here the missing value is more than 100. If we replace them with the mode of rbc, it is not make sence.
So we going to predict the values by linear regression.

Let's do it...............

"""


# ##### Going to replace missing values in categorical columns
# 

# In[9]:


df['appet']=df['appet'].fillna(df['appet'].mode()[0])
df['ane']=df['ane'].fillna(df['ane'].mode()[0])


# 
# ##### Linear regression to find missing value of rbc. For that first we need to encode our categorical data
# 

# #### Encoding catagorical columns as 0 and 1
# #### Step 1: Split missing value rows and non missing value rows
# #### Step 2: Take missing value data set as testing and non missing value as training data sets
# #### Step 3: Do encoding for catagorical variables
# #### Step 4: Do logistic regression 
# #### Step 5: Replace the missing values by the predicted values

# #### 1. Encoding " class" column as 0 and 1

# In[10]:


encoder = LabelEncoder()
df['class']= encoder.fit_transform(df['class'])

df['appet']= encoder.fit_transform(df['appet'])

df['ane']= encoder.fit_transform(df['ane'])


# In[2]:


#seperating missing value rows and non missing rows
train_df = df.dropna() ## data frame without NaN values
test_df = df[df['rbc'].isna()]
train_df


# In[12]:


#encoding rbc column

encoder = LabelEncoder()
train_df['rbc']= encoder.fit_transform(train_df['rbc'])
train_df['rbc'].unique()
train_df


# In[13]:


##moving the dependent column to the last

##for training set

rbc=train_df['rbc']
rbc_train_df=train_df.drop("rbc",axis=1)
rbc_train_df['rbc']=rbc

#for test set

rbc=test_df['rbc']
rbc_test_df=test_df.drop("rbc",axis=1)
rbc_test_df['rbc']=rbc


# In[14]:


#seperating x_train,x_test,y_train,y_test

rbc_x_train = rbc_train_df.iloc[:,:-1]
rbc_y_train =rbc_train_df[["rbc"]]
rbc_x_test = rbc_test_df.iloc[:,:-1]
rbc_y_test = rbc_test_df[["rbc"]]
print(rbc_x_train.shape,"\t",rbc_x_test.shape,"\t",rbc_y_train.shape,"\t",rbc_y_test.shape)


# In[15]:


#predicting missing values of rbc using logistic regression

model = LogisticRegression()
model.fit(rbc_x_train,rbc_y_train)
pred = model.predict(rbc_x_test)
test_df['rbc']=pred


# # Saving the preprocessed dataset

# In[16]:


new_df=pd.concat([train_df,test_df],ignore_index=True) #concatinatiing test and train data previously splited
new_df


# In[17]:


new_df.to_csv("ChronicKidneyDiseaseProcessed.csv")


# ## Prediction

# In[3]:


df = pd.read_csv("ChronicKidneyDiseaseProcessed.csv")
df.head(20)


# In[4]:


X=df.iloc[:,:-1]
Y=df[["class"]]


# ## Without Scaling (vizualization)

# In[5]:


plt.figure(figsize=(10,7))
plt.boxplot(X)
plt.show()


# ## After Scaling Vizualization

# In[6]:


#we are going to scale our data

sc_X = StandardScaler()
X= sc_X.fit_transform(X)


# In[7]:


plt.figure(figsize=(10,7))
plt.boxplot(X)
plt.show()


# # Dimensonality Reduction using PCA 

# In[8]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 17)
X = pca.fit_transform(X)


# In[80]:


explained_ratio= pca.explained_variance_ratio_
np.cumsum(explained_ratio)


# """ We are going to take only 12 features out of 17 features.. because 90% of data is explained by these 13 features. It is enough to build our model """

# In[81]:


X=df.iloc[:,:12]


# In[82]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8)


# In[83]:


x_test.shape


# # Model Building and Prediction

# ## Logistic Regression

# In[84]:


acc = []
prsn = []
f1 = []
rcll = []


# In[85]:


# model building
log_reg_model = LogisticRegression()
log_reg_model.fit(x_train,y_train)
#prediction
log_pr = np.round(log_reg_model.predict(x_test))
log_pr


# In[86]:


#Saving the model

#import pickle
#filename = 'C:/Users/Karthika/project/SavedModels/logistic.pkl'
#pickle.dump(log_reg_model, open(filename, 'wb'))


# #### Accuracy Metrics

# In[87]:


log_acc = np.round(accuracy_score(y_test,log_pr),2)
acc.append(log_acc)

log_prsn = np.round(precision_score(y_test,log_pr),2)
prsn.append(log_prsn)

log_f1 = np.round(f1_score(y_test,log_pr),2)
f1.append(log_f1)

log_rcll = np.round(recall_score(y_test,log_pr),2)
rcll.append(log_rcll)

"Accuracy Score {}    Precision Score {}    F1 Score {}     Recall Score{}".format(log_acc,log_prsn,log_f1,log_rcll)


# ## Support Vector Machine

# In[88]:


svc_clf = SVC(kernel='linear') 
svc_clf.fit(x_train, y_train) 

svc_pr = svc_clf.predict(x_test)


# In[89]:


#saving the model

#filename = 'C:/Users/Karthika/project/SavedModels/svc.pkl'
#pickle.dump(svc_clf, open(filename, 'wb'))


# #### Accuracy metrics

# In[90]:


svm_acc = np.round(accuracy_score(y_test,svc_pr),2)
acc.append(svm_acc)

svm_prsn = np.round(precision_score(y_test,svc_pr),2)
prsn.append(svm_prsn)

svm_f1 = np.round(f1_score(y_test,svc_pr),2)
f1.append(svm_f1)

svm_rcll = np.round(recall_score(y_test,svc_pr),2)
rcll.append(svm_rcll)

"Accuracy Score {}    Precision Score {}    F1 Score {}     Recall Score{}".format(svm_acc,svm_prsn,svm_f1,svm_rcll)


# In[91]:


svc_score = cross_val_score(svc_clf,X,Y,cv=5)
svc_score


# ## K-Nearest Neighbors

# In[92]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn_pr=knn.predict(x_test)


# In[93]:


knn_acc = np.round(accuracy_score(y_test,knn_pr),2)
acc.append(knn_acc)

knn_prsn = np.round(precision_score(y_test,knn_pr),2)
prsn.append(knn_prsn)

knn_f1 = np.round(f1_score(y_test,knn_pr),2)
f1.append(knn_f1)

knn_rcll = np.round(recall_score(y_test,knn_pr),2)
rcll.append(knn_rcll)

"Accuracy Score {}    Precision Score {}    F1 Score {}     Recall Score{}".format(knn_acc,knn_prsn,knn_f1,knn_rcll)


# In[94]:


#Saving the model

#filename = 'C:/Users/Karthika/project/models/knn_model.pkl'
#pickle.dump(knn, open(filename, 'wb'))


# ## Decison Tree

# In[95]:


dsn_tree = DecisionTreeClassifier(criterion="gini",splitter="random",random_state=10)
dsn_tree.fit(x_train,y_train)
dsn_pr = dsn_tree.predict(x_test)


# In[96]:


dsn_acc = np.round(accuracy_score(y_test,dsn_pr),2)
acc.append(dsn_acc)

dsn_prsn = np.round(precision_score(y_test,dsn_pr),2)
prsn.append(dsn_prsn)

dsn_f1 = np.round(f1_score(y_test,dsn_pr),2)
f1.append(dsn_f1)

dsn_rcll = np.round(recall_score(y_test,dsn_pr),2)
rcll.append(dsn_rcll)

"Accuracy Score {}    Precision Score {}    F1 Score {}     Recall Score{}".format(dsn_acc,dsn_prsn,dsn_f1,dsn_rcll)


# In[ ]:


#saving the model

#filename = 'C:/Users/Karthika/project/models/decision_tree.pkl'
#pickle.dump(dsn_tree, open(filename, 'wb'))


# ## Neural Network

# In[104]:


from keras import models,layers


# In[105]:


model = models.Sequential()
model.add(layers.Dense(512,activation='relu',input_dim=12))
model.add(layers.Dense(216,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
#model.add(layers.Dense(84,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


# In[106]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# In[107]:


hist = model.fit(x_train,y_train,epochs = 200,verbose=0)


# In[101]:


plt.plot(range(1,201),hist.history['accuracy'])


# In[102]:


neu_pr = np.round(model.predict(x_test))


# In[103]:


neunet_acc = np.round(accuracy_score(y_test,neu_pr),2)
acc.append(dsn_acc)

neunet_prsn = np.round(precision_score(y_test,neu_pr),2)
prsn.append(neunet_prsn)

neunet_f1 = np.round(f1_score(y_test,neu_pr),2)
f1.append(neunet_f1)

neunet_rcll = np.round(recall_score(y_test,neu_pr),2)
rcll.append(neunet_rcll)

"Accuracy Score {}    Precision Score {}    F1 Score {}     Recall Score{}".format(neunet_acc,neunet_prsn,neunet_f1,neunet_rcll)


# In[64]:


#Saving the model
#model.save('C:/Users/Karthika/flask_demo/SavedModels/DeepLearningModel')


# ## Accuracy, Precision, F1 and Recall Scores of all the models

# In[69]:


Names = ["Logistic Regression", "Support Vector Machine", "Knn", "Decision Tree", "Neural Network"]
scores = {"Accuracy":acc,"Precision Score":prsn,"F1 Score":f1,"Recall":rcll}


# In[108]:


scores = pd.DataFrame(scores,index=Names) 
scores


# In[ ]:




