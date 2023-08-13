import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/python codes/ml/k_mean.csv")
print(data.head())

k=3    #number of clusters
kmean=KMeans(n_clusters=k)

#training the model

kmean=kmean.fit(data)

labels=kmean.labels_   #array that contains cluster number
centroids=kmean.cluster_centers_


# testing data

x_test=[[49.6,67],[27.88,60],[94.65,98],[31.4,-56],[-1.33,5.6],[14.555,-1.22]]


prediction=kmean.predict(x_test)
print(prediction)

colors=['blue','red','green','black']
y=0
for x in labels:
    plt.scatter(data.iloc[y,0],data.iloc[y,1],color=colors[x])
    y+=1

for x in range(k):
    lines=plt.plot(centroids[x,0],centroids[x,1],'kx')
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
title=('number of clusters(K) ={}').format(k)
plt.title(title)
plt.xlabel('V1')
plt.ylabel('V2')
plt.show()
