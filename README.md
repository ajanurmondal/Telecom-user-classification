# Telecom-user-classification
To classify customers according to their internet usage and subscription.

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Importing datasets
df=pd.read_csv('telecom_users.csv')
df.head()
#../input/telecom-data/telecom_users.csv

df.head(20)
df.shape
df.dtypes

# Total charges data type is object, we need to convert into float
df['TotalCharges'].dtypes

df.isnull().sum()
#No null value is present

#categorisation od data
df=df.replace('Yes', 1)
df=df.replace('No',0)

df.head()

pd.set_option('display.max_columns', None)
df.head()

df=df.replace('No internet service', 2)
df=df.replace('No phone service',3)
df.head(5)

df['gender']=df['gender'].astype('category')
df['gender']=df['gender'].cat.codes
#Through categorisation 'Male'=1, 'Female'=0
df.head()
df.isnull().sum()
df['PaymentMethod']=df['PaymentMethod'].astype('category')
df['PaymentMethod']=df['PaymentMethod'].cat.codes
df.head(20)

df=df.drop(['Unnamed: 0','customerID'], axis=1)
df.head()
df['InternetService']=df['InternetService'].astype('category')
df['InternetService']=df['InternetService'].cat.codes
df.head(7)


df['Contract']=df['Contract'].astype('category')
df['Contract']=df['Contract'].cat.codes
df.head()

df['Contract'].value_counts()
df.dtypes

df['MonthlyCharges']=df['MonthlyCharges'].astype(int)
df['TotalCharges']=df['TotalCharges'].astype(str)
df['TotalCharges']=df['TotalCharges'].replace(' ', 0)
df['TotalCharges']=df['TotalCharges'].astype(float)
df['TotalCharges']=df['TotalCharges'].astype(int)

#Creating new dataframe with total carges
df_tot=df['TotalCharges']

#Feature matrics
x=df.drop('Churn', axis=1)
#Response vector
y=df['Churn']

(In this scenario our prime moto is classification of customer base. Linear regression is not an effective solution for classification problem. I will use different classification algorithm like: Logistic regression, K-Nearest Neighbour, Support vector machine, Decision tree etc.)

#Importing SK Learn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
#Normalisation of feature matrics through standard scaler
x=preprocessing.StandardScaler().fit_transform(x)
x[0:5]

#Split the data into train and test data format
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=6)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

## Finding the model accuracy with Logistic Regression
model_logistic=LogisticRegression()
model_logistic.fit(x_train, y_train)
y_prob=model_logistic.predict(x_test)
y_prob
model_logistic.score(x_test, y_test)

from sklearn import metrics
metrics.confusion_matrix(y_test, y_prob)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_prob))

## K-Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=6)
knn_model.fit(x_train, y_train)
y_pred_knn=knn_model.predict(x_test)
y_pred_knn
y_pred_knn_train=knn_model.predict(x_train)
y_pred_knn_train
knn_model.score(x_test, y_test)
print('Accuracy of test data :', metrics.accuracy_score(y_test, y_pred_knn))
print('Accuracy of train data :', metrics.accuracy_score(y_train, y_pred_knn_train))

# Best 'k' value for maximum model accuracy :

k_range=range(1,20)
score=[]
for k in k_range:
    knn_model=KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train, y_train)
    y_pred=knn_model.predict(x_test)
    score.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(k_range, score, color='r')
plt.xlabel('Values for KNN')
plt.ylabel('Model Accuracy in Percentage')

plt.tight_layout
plt.show()
(Maximum accuracy is reached when 'k=14')

## Support Vector Machine
from sklearn.svm import SVC
svm_model=SVC(kernel='linear', gamma=0.1, random_state=5)
svm_model.fit(x_train, y_train)

y_pred_svm=svm_model.predict(x_test)
y_pred_svm
print('Model Accuracy', metrics.accuracy_score(y_test, y_pred_svm))
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_pred_svm))

from sklearn.metrics import jaccard_score, f1_score

print('Jaccard Coefficient:', jaccard_score(y_test, y_pred_svm))
print('F1 SCore: ', f1_score(y_test, y_pred_svm))

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT_model=DecisionTreeClassifier(criterion='entropy', max_depth=6, splitter='random', min_impurity_decrease=0.0)
DT_model.fit(x_train, y_train)

y_pred_dt=DT_model.predict(x_test)
y_pred_dt
print('Model Accuray: ', metrics.accuracy_score(y_test, y_pred_dt))
