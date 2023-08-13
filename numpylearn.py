import numpy as np
#myarr=np.array([3,6,777777,56],np.int8) --generates an error
myarr=np.array([3,6,777777,56],np.int64) 
print(myarr)
#to get an identity matrix
eye=np.eye(4)
print(eye)
one=np.ones(4)
print(one)
one=np.ones((2,3))
print(one)
zero=np.zeros((3,5))
print(zero)
c=np.full((2,3),5)
print(c)
print(c.sum()) # same as np.sum(c)
print(c.prod())
print(c.mean())
print(c.std())
n=np.arange(5)
print(n) #[0,1,2,3,4]
m=np.arange(5,11)
print(m) #[5,6,7,8,9,10]
even=np.arange(0,10,2)
ran=np.random.rand(5)
print(ran) #print 5 random value between 0 and 1
ran1=np.random.rand(2,3)
print(ran1)
ran2=np.random.randint(2,45)
print(ran2)


arr1=np.array([3,4])
arr2=np.array([2,3])
print(10*np.dot(arr1,arr2))
