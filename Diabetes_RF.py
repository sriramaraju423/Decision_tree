#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np


# In[3]:


Diabetes = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 19 - Decision tree & Random forest\Datasets\Diabetes.csv")
Diabetes.head(10)


# In[4]:


Diabetes.shape


# ### EDA

# In[5]:


#Checking for null&NAN values


# In[6]:


Diabetes.isnull().sum()


# In[7]:


#Checking for outliers


# In[8]:


#Drawing boxplot with class variable


# In[9]:


sb.boxplot(x=Diabetes[' Class variable'], y=Diabetes[' Age (years)'])


# In[10]:


# Boxplot results for all col:
# ' Number of times pregnant' = good, ' Plasma glucose concentration' = not good,
#        ' Diastolic blood pressure' = not good, ' Triceps skin fold thickness' = good,
#        ' 2-Hour serum insulin' = not good, ' Body mass index' = not good,
#        ' Diabetes pedigree function' = not good, ' Age (years)' = not good, ' Class variable'


# In[11]:


#We need to do outlier handling


# In[12]:


#Outlier handling


# In[13]:


Q1 = Diabetes.quantile(0.25)
Q3 = Diabetes.quantile(0.75)
IQR = Q3 - Q1 
IQR


# In[14]:


#Even before handling outliers let's try and build model first and then let's handle outliers if the accuracy is so bad


# In[15]:


#There is over fitting problem hence handling outliers


# In[16]:


#In classification models, we need not consider the categorical data i.e. can exclude categorical columns


# In[17]:


col_continous = Diabetes.columns[1:8]
col_continous


# In[18]:


Diabetes_no_outlier = pd.DataFrame(columns=Diabetes.columns)
for i in col_continous:
    min_value = Diabetes[i].min()
    max_value = Diabetes[i].max()
    Diabetes_no_outlier[i]=np.where(Diabetes[i]<(Q1-1.5*IQR)[i],min_value,Diabetes[i])
    Diabetes_no_outlier[i]=np.where(Diabetes[i]>(Q3+1.5*IQR)[i],max_value,Diabetes[i])
Diabetes_no_outlier[' Number of times pregnant'] = Diabetes[' Number of times pregnant']
Diabetes_no_outlier[' Class variable'] = Diabetes[' Class variable']
Diabetes_no_outlier


# In[19]:


#Building model with 'Diabetes_no_outlier' dataset


# In[20]:


#Building model


# In[21]:


#Dividing data into train and test


# In[22]:


Diabetes_train,Diabetes_test = train_test_split(Diabetes,test_size=0.2)


# In[23]:


Diabetes_col = Diabetes.columns
predictors = Diabetes_train[Diabetes_col[:8]]
target = Diabetes_train[Diabetes_col[8]]
# Diabetes_col = Diabetes_no_outlier_feature_4.columns
# predictors = Diabetes_train[Diabetes_col[:4]]
# target = Diabetes_train[Diabetes_col[4]]


# In[24]:


rf_model = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion='entropy')


# In[25]:


rf_model.fit(predictors,target)


# In[26]:


pred_train = rf_model.predict(predictors)


# In[27]:


accuracy = np.mean(pred_train==target)
accuracy


# In[28]:


#Accuracy of testing data


# In[29]:


predictors_test = Diabetes_test[Diabetes_col[:8]]
target_test = Diabetes_test[Diabetes_col[8]]
# predictors_test = Diabetes_test[Diabetes_col[:4]]
# target_test = Diabetes_test[Diabetes_col[4]]


# In[30]:


pred_test = rf_model.predict(predictors_test)


# In[31]:


accuracy_test = np.mean(pred_test==target_test)
accuracy_test


# In[32]:


#Getting the 'feature_importances' of this model and then try and remove unimportant columns and check the accuracy


# In[33]:


feature_importances = rf_model.feature_importances_
feature_importances


# In[34]:


indices = np.argsort(-feature_importances)
indices


# In[35]:


Diabetes_col[indices]


# In[36]:


#After watching the scores let's keep the first 4 and ignore the rest


# In[37]:


Diabetes_no_outlier_feature_4_col_names = Diabetes_col[indices][:4]
Diabetes_no_outlier_feature_4_col_names
Diabetes_no_outlier_feature_4 = Diabetes_no_outlier[Diabetes_no_outlier_feature_4_col_names]
#Adding output variable
Diabetes_no_outlier_feature_4[' Class variable'] = Diabetes_no_outlier[' Class variable']
Diabetes_no_outlier_feature_4


# In[38]:


#Now let's build model with 'Diabetes_no_outlier_feature_4'


# In[39]:


# Doing transformations on Diabetes_no_outlier


# In[40]:


Diabetes_no_outlier_trans = Diabetes_no_outlier
Diabetes_no_outlier_trans[Diabetes_no_outlier_feature_4_col_names] = np.log(Diabetes_no_outlier_trans[Diabetes_no_outlier_feature_4_col_names])
Diabetes_no_outlier_trans


# In[41]:


Diabetes_no_outlier_trans.replace([np.inf, -np.inf], np.nan,inplace=True)


# In[42]:


Diabetes_no_outlier_trans.isnull().sum()


# In[43]:


np.where(np.isnan(Diabetes_no_outlier_trans[' Plasma glucose concentration']))


# In[44]:


Diabetes_no_outlier_trans.fillna(Diabetes_no_outlier_trans.mean(),inplace=True)


# In[45]:


Diabetes_no_outlier_trans.isnull().sum()


# In[46]:


Diabetes_no_outlier_trans.loc[182]


# In[47]:


Diabetes_no_outlier_trans.shape


# In[48]:


#Now let's try and build model with transformed data


# In[49]:


#Pretty absurb accuracy, high overfitting model. Let's try and do the following steps


# In[50]:


#1.Build model with more no of trees and check accuracy. try 1: 25 -> Accuracy: 75; try 2: 50 -> Accuracy: 76; try 3: 5 -> Accuracy: 70. Not much of change so it doesn't make sense to keep on increasing trees because there would be many overlapps and result doesn't change much
#2.Let me try and remove outliers from dataset. try 1-> Accuracy: 0.7142857142857143; try 2: 20 -> Accuracy: 0.7532467532467533; try 3: 50 -> Accuracy: 0.7597402597402597
#3.I read a blog which says try and use only the important 'feature_importances' columns. try 1-> 15 -> Accuracy: 0.7597402597402597; try 2-> 50 -> Accuracy: 0.7922077922077922. Still negative with feature importance columns
#4. Tranformations: I read in some blog that: 'The way Random Forests are built is invariant to monotonic transformations of the independent variables. Splits will be completely analogous. If you are just aiming for accuracy you will not see any improvement in it'. Ideally it doesn't make any sense do tranformation because whole essence of random forest is it is rule based and hence explinatory but Still wanna give it a try. Try 1 -> 50 -> 0.7727272727272727


# #### Conclusion: I am pretty sure that after trying all the possible ways to increase accuracy, random forest does what it best does when we have a larger dataset. And it's a major contributer to ensemble technique when opted rather than trying solve any problem by just using only random forest

# #### Queries: 
# 1)Plz do let me know if there is any other way using which i can increase the accuracy.
# 2)One of the added advatage of random forest is 'Effective for missing values', but as part of EDA i am forced to do imputation of Nan values with mean values. Without which model is not taking fit data. Then how are we supposed to acheive this?
