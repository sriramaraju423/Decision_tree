#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# In[2]:


Company_Data = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 19 - Decision tree & Random forest\Datasets\Company_Data.csv")
Company_Data.head(10)


# In[161]:


type(Company_Data)


# In[ ]:


Company_Data.shape


# In[ ]:


#Checking for missing values for imputation


# In[ ]:


Company_Data.isnull().sum()


# In[ ]:


#Let's transform the >2 categorical data using label encoder into labels


# In[ ]:


label_col = ['ShelveLoc',"Urban","US"]


# In[ ]:


for i in label_col:
    Company_Data[i] = LabelEncoder().fit_transform(Company_Data[i])
Company_Data.head(10)


# In[ ]:


Company_Data.iloc[27,:]


# In[ ]:


#Now we have to convert target col 'Sales' into categorical


# In[ ]:


Company_Data.iloc[:,0].describe()


# In[ ]:


bins = [0,5,10,15,20]


# In[ ]:


categories = pd.cut(Company_Data.iloc[:,0],bins,labels=[0,1,2,3],include_lowest =True)
categories


# In[ ]:


type(categories)


# In[ ]:


categories.loc[174]


# In[ ]:


categories = categories.to_frame()


# In[ ]:


categories.iloc[174,:]


# In[ ]:


np.where(np.isnan(categories))


# In[ ]:


categories.loc[27]


# In[ ]:


categories.value_counts()


# In[ ]:


Company_Data['Sales'] = categories
Company_Data


# In[201]:


Company_Data.describe()


# In[202]:


#Checking for outliers & replacing them with min and max values of that particular column


# In[160]:


#In classification models, we need not consider the categorical data i.e. can exclude categorical columns


# In[182]:


col_names_continous = ['CompPrice','Income','Advertising','Population','Price','Age','Education']


# In[185]:


Q1 = Company_Data.quantile(0.25)
Q3 = Company_Data.quantile(0.75)
IQR = Q3 - Q1
IQR


# In[200]:


for i in col_names_continous:
    min_value = Company_Data[i].min()
    max_value = Company_Data[i].max()
    Company_Data[i]=np.where(Company_Data[i]<(Q1-1.5*IQR)[i],min_value,Company_Data[i])
    Company_Data[i]=np.where(Company_Data[i]>(Q3+1.5*IQR)[i],max_value,Company_Data[i])
Company_Data


# In[ ]:


#Now we have the transformed data on which we can perform the required decision tree technique


# In[203]:


col_names = Company_Data.columns


# In[204]:


predictors = Company_Data[col_names[1:]]
target = Company_Data[col_names[0]]


# In[205]:


#Let's split the data for training and testing


# In[221]:


train,test = train_test_split(Company_Data,test_size=0.3)


# In[222]:


predictors_train = train[col_names[1:]]
target_train = train[col_names[0]]


# In[223]:


predictors_train.isnull().sum()


# In[224]:


type(predictors_train)


# In[225]:


np.where(np.isnan(target_train))


# In[211]:


#Now let's build the model using decisiontree classifier


# In[226]:


company_train_model = DecisionTreeClassifier(criterion='entropy')


# In[227]:


company_train_model.fit(predictors_train,target_train)


# In[228]:


#Predicting the train data and checking accuracy


# In[229]:


pred_train = company_train_model.predict(predictors_train)


# In[230]:


accuracy = np.mean(pred_train==target_train)
accuracy


# #### Train accuracy = 1.0

# In[217]:


#Now let's try and find out test accuracy


# In[231]:


predictors_test = test[col_names[1:]]
target_test = test[col_names[0]]


# In[232]:


pred_test = company_train_model.predict(predictors_test)


# In[233]:


accuracy_test = np.mean(pred_test==target_test)
accuracy_test


# #### Test accuracny = 0.6875

# In[ ]:


#There is overfitting problem we may need to try and fix this


# #### Accuracy after outlier treatment is 0.625

# In[234]:


#Conclusion for overfitting: I am quite not sure what is the root cause for this overfitting problem, i guess this dataset doesn't quite fit for decison tree classification model.
#Steps i tried: 1)splitting 80 20 2)splitting 70 30 3)Outlier treatment: replaced outliers with max and min values
#My solution would be using this model as one the ensemble models. Will try this at later point of time when working on ensemble techniques


# In[235]:


#Problem statement: To find out 'attributes contributing to high sale'


# In[261]:


feature_importances = company_train_model.feature_importances_
feature_importances


# In[265]:


indices = np.argsort(feature_importances)
indices


# In[266]:


#By default argsort give you the elements index in ascending order to make it decending you just need to negate the input


# In[267]:


indices = np.argsort(-feature_importances)
indices


# In[268]:


col_names[indices]


# In[ ]:


#Conclusion: THis is the order of importance of attributes contributing to high sales

