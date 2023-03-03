#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install imbalanced-learn


# In[2]:


import pandas as pd
import numpy as np
import imblearn


# In[3]:


data=pd.read_csv('train.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


import seaborn as sns


# In[7]:


sns.countplot(data.target,data=data,)


# In[8]:


data.groupby('target')['id'].count()/len(data)


# In[9]:


data.describe()


# In[10]:


x=data.iloc[:,2:]
y=data.loc[:,'target']


# In[11]:


cat_col=[i for i in range(len(x.columns)) if 'cat' in x.columns[i]]


# In[12]:


cat_col


# In[13]:


from catboost import CatBoostClassifier


# In[14]:


cbc=CatBoostClassifier(iterations=300,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=13)


# In[17]:


cbc.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


y_pred=cbc.predict(x_test)


# In[20]:


print(classification_report(y_test,y_pred))


# In[21]:


print(confusion_matrix(y_test,y_pred))


# In[22]:


from imblearn.under_sampling import RandomUnderSampler


# In[23]:


rus=RandomUnderSampler(random_state=13)


# In[24]:


x_rus,y_rus=rus.fit_resample(x,y)


# In[25]:


x_rus.shape


# In[26]:


y_rus.shape


# In[27]:


y_rus.value_counts()


# In[28]:


y.value_counts()


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x_rus,y_rus,random_state=13)


# In[30]:


cbcu=CatBoostClassifier(iterations=1000,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[31]:


cbcu.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[32]:


y_pred=cbcu.predict(x_test)


# In[33]:


print(classification_report(y_test,y_pred))


# In[34]:


print(confusion_matrix(y_test,y_pred))


# In[35]:


from imblearn.over_sampling import RandomOverSampler


# In[36]:


ros=RandomOverSampler(random_state=13)


# In[37]:


x_ros,y_ros=ros.fit_resample(x,y)


# In[38]:


x_ros.shape


# In[39]:


y_ros.shape


# In[40]:


y_ros.value_counts()


# In[41]:


y.value_counts()


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x_ros,y_ros,random_state=13)


# In[43]:


cbco=CatBoostClassifier(iterations=1000,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[44]:


cbco.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[45]:


y_pred=cbco.predict(x_test)


# In[46]:


print(classification_report(y_test,y_pred))


# In[47]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:




