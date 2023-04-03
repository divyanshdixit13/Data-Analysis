#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install seaborn==0.9.0 matplotlib==3.5.0 scikit-Learn==0.20.1')


# In[4]:


get_ipython().system('pip install -U scikit-learn')


# In[5]:


get_ipython().system('pip install -U seaborn')


# In[6]:


get_ipython().system('pip install -U matplotlib')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


URL="C:\\Users\\DELL\\Downloads\\CERA DA\\kc_house_data.csv"
df= pd.read_csv(URL)


# In[10]:


df.head()


# In[11]:


# Display the Data types of each column(using ctrl+/)
df.dtypes


# In[33]:


# Data Wrangling
df.drop(id, axis=1, inplace=True)
df.describe()


# In[32]:


df.columns


# In[35]:


df.describe()


# In[39]:


print("Number of Nan values for column bedrooms:", df['bedrooms'].isnull().sum())
print("Number of Nan values for column bathrooms:", df['bathrooms'].isnull().sum())


# In[41]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[42]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[43]:


print("Number of Nan values for column bedrooms:", df['bedrooms'].isnull().sum())
print("Number of Nan values for column bathrooms:", df['bathrooms'].isnull().sum())


# In[44]:


#Exploratory Data Analysis


# In[47]:


floor_count= df['floors'].value_counts().to_frame()
print(floor_count)


# In[52]:


sns.boxplot(x='waterfront', y='price',data=df)


# In[53]:


# Using Regplot


# In[54]:


sns.regplot(x='sqft_above',y='price',data=df)


# In[55]:


df.corr()['price'].sort_values()


# In[56]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[58]:


X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[59]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X=[['features']]
Y=['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[ ]:




