import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
# gathering of data
data=pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\my_doc\\housing.csv")

y=np.array(data['price'])
x=np.array(data['area'])
x=x.reshape(len(x),1)
y=y.reshape(len(y),1)

#preparing/splitting the data
X_train = x[:-250]
X_test=x[-250:]
Y_train = y[:-250]
Y_test=y[-250:]

regr=linear_model.LinearRegression()
regr.fit(X_train,Y_train)


plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,regr.predict(X_test),color='red',linewidth=3)
plt.title('Test Data')
plt.ylabel('size')
plt.xlabel('Price')
plt.show()






