#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn==0.9.0 matplotlib==3.5.0 scikit-Learn==0.20.1')


# In[2]:


get_ipython().system('pip install -U scikit-learn')


# In[3]:


get_ipython().system('pip install -U seaborn')


# In[4]:


get_ipython().system('pip install -U matplotlib')


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


URL="C:\\Users\\DELL\\Downloads\\CERA DA\\kc_house_data.csv"
df= pd.read_csv(URL)


# In[8]:


df.head()


# In[9]:


# Display the Data types of each column(using ctrl+/)
df.dtypes


# In[10]:


# Data Wrangling
df.drop(id, axis=1, inplace=True)
df.describe()


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


print("Number of Nan values for column bedrooms:", df['bedrooms'].isnull().sum())
print("Number of Nan values for column bathrooms:", df['bathrooms'].isnull().sum())


# In[ ]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[ ]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[ ]:


print("Number of Nan values for column bedrooms:", df['bedrooms'].isnull().sum())
print("Number of Nan values for column bathrooms:", df['bathrooms'].isnull().sum())


# In[ ]:


#Exploratory Data Analysis


# In[39]:


floor_count= df['floors'].value_counts().to_frame()
print(floor_count)


# In[40]:


sns.boxplot(x='waterfront', y='price',data=df)


# In[ ]:


# Using Regplot


# In[41]:


sns.regplot(x='sqft_above',y='price',data=df)


# In[ ]:


df.corr()['price'].sort_values()


# In[ ]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[11]:


# Fitting Linear Regression Model

X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[30]:


# Using pipelines and find R^2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
X = df[features]
y = df['price']
pipe.fit(X,y)
y_pred = pipe.predict(X)
r2_score = pipe.score(X, y)
print(r2_score)


# In[31]:


#Model Evaluation & Refinement
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[32]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[33]:


from sklearn.linear_model import Ridge


# In[34]:


RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(X,Y)
yhat=RidgeModel.predict(X)


# In[36]:


RidgeModel.score(X,Y)


# In[38]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Create polynomial features
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Fit Ridge regression model with regularization parameter 0.1
ridge = Ridge(alpha=0.1)
ridge.fit(x_train_poly, y_train)

# Predict on test data and calculate R^2
y_test_pred = ridge.predict(x_test_poly)
r2 = r2_score(y_test, y_test_pred)

print("R^2 on test data with second order polynomial features and Ridge regression:", r2)







# In[ ]:




