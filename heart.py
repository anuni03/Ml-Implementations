import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
#gathering of data
data=pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/python codes/ml/heartrate.csv")

y=data.target.values
x_data=data.drop(['target'],axis=1) #drop a column

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#Preparing of data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#logistic regression

log=LogisticRegression()
log.fit(x_train,y_train) #training of  model
print('Test accuracy of the Logistic regression: {}'.format(log.score(x_test,y_test)*100))

#KNN

scorelist=[]
for i in range(1,20):
    knn=KNeighborsClassifier(i)
    knn.fit(x_train,y_train)
    prediction=knn.predict(x_test)
    scorelist.append(knn.score(x_test,y_test)*100)
print('Test accuracy of the KNN: {}'.format(max(scorelist)))
#plt.plot(range(1,20),scorelist)
#plt.xlabel('K value')
#plt.ylabel('Score')
#plt.show()

#support vector machine
svm=SVC(random_state=1)
svm.fit(x_train,y_train)
print('Test accuracy of the Support Vector Machine: {}'.format(svm.score(x_test,y_test)*100))

#Naive Bayes Algorithm
nb=GaussianNB()
nb.fit(x_train,y_train)
print('Test accuracy of Naive Bayes: {}'.format(nb.score(x_test,y_test)*100))

#Decision Tree
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
print('Test accuracy of Decision Tree Classifier: {}'.format(dtc.score(x_test,y_test)*100))

#Random Forest
rf=RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)
print('Test accuracy of random forest : {}'.format(rf.score(x_test,y_test)*100))

methods=['Logistic Regression','KNN','Support Vector Machine','Naive Bayes','Decision tree','Random Forest']
accuracy=[62.29,72.13,68.85,85.24,77.04,85.24]
color=['purple','green','orange','magenta','#CFC60E','#0FBBAE']
sb.set_style('whitegrid')
plt.figure(figsize=(16,5))
plt.ylabel("Accuracy(%)")
plt.xlabel("Algorithm")
sb.barplot(x=methods,y=accuracy,palette=color)
plt.show()
