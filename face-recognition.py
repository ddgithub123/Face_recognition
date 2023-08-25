#importing required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

cnames=['Outlook','Temperature','Humidity','Wind','Played football(yes/no)']
#loading datasets
fball=pd.read_csv(r"C:\Users\HP\Desktop\sample football data.csv",header=None,names=cnames)
fball.head()

#splitting dataset in features and target variable
feature_cols=['Outlook','Temperature','Humidity','Wind','Played football(yes/no)']
X=fball[feature_cols]
y=fball.label

#training and test sets
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print(len(x_train)," ", len(y_train))#the length of the training data
print(len(x_test)," ",len(y_test))#the length of the testing sample
#more the tes_size , more data can be used for testing

#create decision tree criterion entrpy used for information gain
clf=DecisionTreeClassifier(criterion="entropy",max_depth=3)
#Train decision classifier
clf=clf.fit(x_train,y_train)
#predict the response for testdatasets
y_pred=clf.predict(x_test)

#Accuracy
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))