#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sb


# In[83]:


Fraud_check = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 19 - Decision tree & Random forest\Datasets\Fraud_check.csv")
Fraud_check.head(10)


# ### EDA

# In[84]:


Fraud_check.shape


# In[85]:


Fraud_check.isnull().sum()


# In[86]:


Fraud_check.describe()


# In[87]:


#It's asked -'treating those who have taxable_income <= 30000 as "Risky" and others are "Good"'


# In[88]:


#So converting 'Taxable.Income' column to 2 categorical


# In[89]:


bins = [0,30000,100000]


# In[90]:


Fraud_check['Taxable.Income'] = pd.cut(Fraud_check['Taxable.Income'],bins,labels=['Risky','Good'],include_lowest=True)
Fraud_check


# In[91]:


Fraud_check['Taxable.Income'].value_counts()


# In[92]:


#Now let's convert predictor columns as labels


# In[93]:


label_col = ['Undergrad',"Marital.Status","Urban"]


# In[94]:


for i in label_col:
    Fraud_check[i] = LabelEncoder().fit_transform(Fraud_check[i])
Fraud_check.head(10)    


# In[95]:


#Now the dataset is good to build the model


# In[96]:


#Build model


# In[97]:


#Splitting data into train and test


# In[98]:


Fraud_train,Fraud_test = train_test_split(Fraud_check,test_size=0.2)


# In[99]:


Fraud_train.shape


# In[100]:


col_names = list(Fraud_check.columns)
col_names.remove('Taxable.Income')
col_names


# In[101]:


predictors = Fraud_train[col_names]
target = Fraud_train['Taxable.Income']


# In[102]:


Fraud_model = DecisionTreeClassifier(criterion='entropy')


# In[103]:


Fraud_model.fit(predictors,target)


# In[104]:


pred = Fraud_model.predict(predictors)


# In[105]:


accuracy = np.mean(pred==target)
accuracy


# In[106]:


#test accuracy


# In[107]:


predictors_test = Fraud_test[col_names]
target_test = Fraud_test['Taxable.Income']


# In[108]:


pred = Fraud_model.predict(predictors_test)


# In[109]:


accuracy_test = np.mean(pred==target_test)
accuracy_test


# In[110]:


Fraud_model.feature_importances_


# In[111]:


#Now let's check for ouliers


# In[112]:


sb.boxplot(x=Fraud_check['Taxable.Income'],y=Fraud_check['City.Population'])


# In[113]:


sb.boxplot(x=Fraud_check['Taxable.Income'],y=Fraud_check['Work.Experience'])


# In[114]:


#It's quite clear there are no outliers


# #### Conclusion: This is the best accuracy that i can get. There is overfitting problem still existing. i suppose both decison tree and random forests does there best when it's in amalgamation with other models in ensemble techniques

# ##### Queries:
# 1)Do let me know if i can do any kind of EDA to avoid this overfitting problem. If not i will be trying this dataset in ensemble technique and try and see if i can build accurate model there.    
