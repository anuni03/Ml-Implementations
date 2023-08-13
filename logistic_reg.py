#Train a logistic regression classifier to predict whether a flower is iris virginica or not



from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris=datasets.load_iris()
X = iris["data"][:,3:]
y=(iris["target"] == 2).astype(int)
#print(y)

#Train A Logistic Regression Classifier
clf=LogisticRegression()
clf.fit(X,y)
example = clf.predict(([[1.6]])) #here it tells whether its true or false
print(example)

#Using matplotlib to plot the visualization
X_new=np.linspace(0,3,1000).reshape(-1,1)
#print(X_new)
y_prob = clf.predict_proba(X_new)  #here we get probability
plt.plot(X_new,y_prob[:,1],"g-",label="virginica")
plt.show()


#print(list(iris.keys()))
#print(iris['target'])

