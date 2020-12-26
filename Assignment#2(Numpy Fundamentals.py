#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[3]:


import numpy as np
arr1 = np.arange(0,10).reshape(2,5)
arr1


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[9]:


arr1 = np.arange(0,10).reshape(2,5)
arr2 = np.ones((2,5))
np.vstack((arr1,arr2)).astype(int)


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[10]:


arr1 = np.arange(0,10).reshape(2,5)
arr2 = np.ones((2,5))
np.hstack((arr1,arr2)).astype(int)


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[12]:


arr1.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[16]:


arr3 = np.arange(0,15).reshape(3,5)
arr3.ravel()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[20]:


arr4 = np.arange(0,15)
arr5 = arr4.reshape(3,5)
arr5


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[23]:


arr6 = np.random.randint(1,7, size=(5, 5))
print("array 5x5 =", arr6)
print("Square of array =", np.square(arr6))


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[24]:


arr7 = np.random.randint(2,8, size = (5, 6))
print("array 5x6 =", arr7)
print("Mean of array =", np.mean(arr7))


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[30]:


sd = np.std(arr7)
print("Standard Deviation:", np.round(sd,3) )


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[31]:


print("Median: ", np.median(arr7))


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[ ]:


print("Transpose: ", np.transpose(arr7))


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[33]:


arr8 = np.random.randint(5,9, size = (4, 4))
print(arr8)
print("sum of diagonal elements: ", np.trace(arr8))


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[35]:


determinant = np.linalg.det(arr8)
print("determinant: ", np.round(determinant,3))


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[37]:


arr9 = np.random.randint(0,100,size=(4, 4))
print(arr9)
print("5th percentile : ", np.percentile(arr9, 5))
print("95th percentile : ", np.percentile(arr9, 95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[38]:


array_null=np.array([0,2,3,4,5,6,7])
array_null = array_null.astype('float')
array_null[2]=np.NaN
print(array_null)
array_sum = np.sum(array_null)
array_has_nan = np.isnan(array_sum)
print(array_has_nan)


# In[ ]:




