#!/usr/bin/env python
# coding: utf-8

# In[23]:


#importing libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


#importing dataset

dataset = pd.read_csv("E:/JP_ML/aml_project/Dataset/Demo3.csv")


# In[25]:


#showing first 5 rows

dataset.head()


# In[26]:


#showing last 5 rows

dataset.tail()


# In[27]:


dataset.info()


# In[28]:


dataset.describe()


# In[29]:


#shape of the dataframe (row,column)

dataset.shape


# In[30]:


#fixed row-column which are useful for the algorithm

x = dataset.iloc[:,[3,4]].values


# In[31]:


#showing dataframe

x


# In[38]:


#plotting in 2D view

plt.scatter(x[:,0],x[:,1], s=5, c='black')


# In[39]:


#plotting using seasborn

sns.pairplot(dataset.iloc[:,[3,4]])


# In[40]:


#Scaling the fields if necessary

from sklearn.preprocessing import StandardScaler
# x = dataset.iloc[:,[1,5]].values         #if we need any special column for scaling 
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


# In[41]:


#importing K-Means clustering

from sklearn.cluster import KMeans


# In[42]:


#elbow method applied

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[43]:


#plotting of elbow method

plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Square")
plt.show()


# In[52]:


#importing DBSCAN algorithm

from sklearn.cluster import DBSCAN


# In[56]:


#value set and fit the model

dbscan = DBSCAN(eps=0.25, min_samples=5)
labels = dbscan.fit_predict(x)


# In[57]:


#determining the number of clusters and outlier

np.unique(labels)


# In[60]:


#plotting the final result

plt.scatter(x[labels == -1,0],x[labels == -1,1], s=10, c='black')    #outlier
plt.scatter(x[labels == 0,0],x[labels == 0,1], s=10, c='red')        #class 1
plt.scatter(x[labels == 1,0],x[labels == 1,1], s=10, c='blue')       #class 2
plt.scatter(x[labels == 2,0],x[labels == 2,1], s=10, c='green')      #class 3
plt.scatter(x[labels == 3,0],x[labels == 3,1], s=10, c='cyan')       #class 4
plt.scatter(x[labels == 4,0],x[labels == 4,1], s=10, c='brown')      #class 5

plt.title("DBSCAN Clustering")
plt.xlabel("")
plt.ylabel("")
plt.show()


# In[ ]:





# In[ ]:




