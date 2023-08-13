from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(iris.data,iris.target)
a=knn.predict([[4,5,6,7]])
print(iris.target_names[a])
