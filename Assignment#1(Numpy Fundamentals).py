#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[5]:


nulvec = np.zeros(10)
nulvec


# 3. Create a vector with values ranging from 10 to 49

# In[6]:


rangevec = np.arange(10,50)
rangevec


# 4. Find the shape of previous array in question 3

# In[7]:


rangevec.shape


# 5. Print the type of the previous array in question 3

# In[9]:


rangevec.dtype


# 6. Print the numpy version and the configuration
# 

# In[12]:


print(np.__version__, np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[13]:


rangevec.ndim


# 8. Create a boolean array with all the True values

# In[15]:


boolarray = np.array(range(1, 10),dtype="bool" )
boolarray


# 9. Create a two dimensional array
# 
# 
# 

# In[21]:


aray2d = np.arange(0,4)
aray2d.reshape(2,2)


# 10. Create a three dimensional array
# 
# 

# In[25]:


aray3d = np.arange(0,27).reshape(3,3,3)
aray3d


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[29]:


print ( "Stright = " ,rangevec)
print ( " reversed = ", rangevec[::-1])


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[30]:


nulvec[5] = 1
nulvec


# 13. Create a 3x3 identity matrix

# In[33]:


idenmat = np.identity(3) 
idenmat


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[34]:


arr = np.array([1, 2, 3, 4, 5])
arrfloat = arr.astype(np.float32)
arrfloat


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[35]:


arr1 = np.array([[1., 2., 3.],
                [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],
                [7., 2., 12.]])
np.multiply(arr1,arr2)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[36]:


arr1 = np.array([[1., 2., 3.],
                [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],
                [7., 2., 12.]])
print("arr1 > arr2") 
print(np.greater(arr1,arr2)) 
  
print("arr1 >= arr2") 
print(np.greater_equal(arr1,arr2)) 
  
print("arr1 < arr2") 
print(np.less(arr1,arr2)) 
  
print("arr1 <= arr2") 
print(np.less_equal(arr1,arr2)) 


# 17. Extract all odd numbers from arr with values(0-9)

# In[43]:


arayrange = np.arange(0,10)
oddarray = arayrange[arayrange % 2 == 1]
oddarray


# 18. Replace all odd numbers to -1 from previous array

# In[45]:


arayrange[arayrange % 2 == 1] = -1
arayrange


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[51]:


arayrange[5:9]=12
arayrange


# 20. Create a 2d array with 1 on the border and 0 inside

# In[52]:


arr = np.ones((3, 4))
arr[1:-1, 1:-1] = 0
arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[53]:


arr2d = np.array([[1, 2, 3],
                  [4, 5, 6], 
                  [7, 8, 9]])
arr2d[1, 1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[57]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0][0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[59]:


aray2d = np.arange(9).reshape(3, 3)
arr2dslice = aray2d[0]
arr2dslice


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[61]:


aray2d[1][1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[64]:


aray2d[0:-1, -1]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[72]:


from numpy import random
arrayrandom = random.randint(100, size=(100)).reshape(10,10)
print (arrayrandom)
maxrand = np.amax(arrayrandom)
maxrand


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[73]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a, b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[74]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a == b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[82]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != 'Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[83]:


data[ (names == 'Bob') | (names == 'Will')]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[85]:


randarr2d = np.random.uniform(1,15, size=(5,3))
randarr2d


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[86]:


randarr3d = np.random.uniform(1,16, size=(2,2,4))
randarr3d


# 33. Swap axes of the array you created in Question 32

# In[89]:


np.swapaxes(randarr3d,1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[95]:


array10 = np.arange(10)
arraysqrt = np.sqrt(array10)
arraysqrtrep = np.where(arraysqrt < 0.5 , 0, arraysqrt)
arraysqrtrep


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[104]:


arrayrandom01 = random.randint(100,size=(12))
arrayrandom02 = random.randint(100,size=(12))
print ("array01 = ", arrayrandom01)
print ("array02 = ", arrayrandom02)
np.maximum(arrayrandom01,arrayrandom02)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[105]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[108]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
np.setdiff1d(a, b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[112]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray[:, 1] = 10
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[113]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x, y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[114]:


arrayrandom03 = random.randint(100,size=(20))
np.cumsum(arrayrandom03)

