import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
#print(diabetes.keys())
#print(diabetes.DESCR) - to know all features

#diabetes_X=diabetes.data ---this is to get data  from diabetes
diabetes_X=np.array([[1],[2],[3]])
#diabetes_X_train=diabetes_X[:-30]
diabetes_X_train=diabetes_X
#diabetes_X_test=diabetes_X[-30:]
diabetes_X_test=diabetes_X

#diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_train=np.array([3,2,4])
#diabetes_Y_test=diabetes.target[-30:]
diabetes_Y_test=np.array([3,2,4])

model=linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean squared error is: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))

print("Weights: ",model.coef_)
print("Intercept: ",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)

plt.show()
