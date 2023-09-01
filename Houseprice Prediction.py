#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[4]:


df=pd.read_csv("C:/Users/DHANALAKSHMI/Downloads/archive/Housing.csv")


# In[5]:


df.head()


# In[95]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# Data Cleaning

# In[9]:


# checking Null values
df.isnull().sum()*100/df.shape[0]


# In[10]:


#outlier analysis
fig,axs=plt.subplots(2,3,figsize=(10,5))
plt1=sns.boxplot(df['price'],ax=axs[0,0])
plt2=sns.boxplot(df['area'],ax=axs[0,1])
plt3=sns.boxplot(df['bedrooms'],ax=axs[0,2])
plt1=sns.boxplot(df['bathrooms'],ax=axs[1,0])
plt1=sns.boxplot(df['stories'],ax=axs[1,1])
plt1=sns.boxplot(df['parking'],ax=axs[1,2])
plt.tight_layout()


# In[11]:


#outlier treatment for price
plt.boxplot(df.price)
Q1=df.price.quantile(0.25)
Q3=df.price.quantile(0.75)
IQR=Q3-Q1
df=df[(df.price>=Q1-1.5*IQR)&(df.price<=Q3+1.5*IQR)]


# In[12]:


#outlier treatment for area
plt.boxplot(df.area)
Q1=df.area.quantile(0.25)
Q3=df.area.quantile(0.75)
IQR=Q3-Q1
df=df[(df.area>=Q1-1.5*IQR)&(df.area<=Q3+1.5*IQR)]


# In[13]:


#outlier analysis
fig,axs=plt.subplots(2,3,figsize=(10,5))
plt1=sns.boxplot(df['price'],ax=axs[0,0])
plt2=sns.boxplot(df['area'],ax=axs[0,1])
plt3=sns.boxplot(df['bedrooms'],ax=axs[0,2])
plt1=sns.boxplot(df['bathrooms'],ax=axs[1,0])
plt1=sns.boxplot(df['stories'],ax=axs[1,1])
plt1=sns.boxplot(df['parking'],ax=axs[1,2])
plt.tight_layout()


# # Exploratory Data Analytics

# In[14]:


#visualising Numeric variables
sns.pairplot(df)
plt.show()


# In[16]:


plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x='mainroad',y='price',data=df)
plt.subplot(2,3,2)
sns.boxplot(x='guestroom',y='price',data=df)
plt.subplot(2,3,3)
sns.boxplot(x='basement',y='price',data=df)
plt.subplot(2,3,4)
sns.boxplot(x='hotwaterheating',y='price',data=df)
plt.subplot(2,3,5)
sns.boxplot(x='airconditioning',y='price',data=df)
plt.subplot(2,3,6)
sns.boxplot(x='furnishingstatus',y='price',data=df)
plt.show()


# In[17]:


plt.figure(figsize=(10,5))
sns.boxplot(x='furnishingstatus',y='price',hue='airconditioning',data=df)
plt.show()


# In[84]:


plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[87]:


x=df[['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea']]
y=df['price']


# In[88]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=100)


# In[89]:


from sklearn.linear_model import LinearRegression


# In[90]:


ln=LinearRegression()
ln.fit(x_train,y_train)


# In[91]:


coeff=pd.DataFrame(ln.coef_,x.columns,columns=['coefficient'])
coeff


# In[92]:


prediction=ln.predict(x_test)
plt.scatter(y_test,prediction)


# In[94]:


plt.scatter(y_test,prediction)
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("Actual prices vs predicted prices")
plt.show()


# In[ ]:




