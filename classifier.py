#loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading dataset
iris=datasets.load_iris()
#print(iris.DESCR) ---> to print the description

#printing features
features=iris.data
labels=iris.target
print(features[0],labels[0])

#Training the classifier

clf=KNeighborsClassifier()
clf.fit(features,labels)

preds=clf.predict([[9,1,1,1]])
print(preds)

