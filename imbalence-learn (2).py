#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import imblearn
import random
from imblearn.under_sampling import TomekLinks


# In[2]:


n=10000
skip=sorted(random.sample(range(1,595212),595212-n))
data=pd.read_csv('train.csv',skiprows=skip)


# In[3]:


data


# In[4]:


data.groupby('target')['id'].count()/len(data)


# In[5]:


x=data.iloc[:,2:]
y=data.loc[:,'target']


# In[6]:


cat_col=[i for i in range(len(x.columns)) if 'cat' in x.columns[i]]


# In[7]:


cat_col


# In[8]:


from catboost import CatBoostClassifier


# In[9]:


cbc=CatBoostClassifier(iterations=300,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=13)


# In[12]:


cbc.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[13]:


from sklearn.metrics import classification_report,confusion_matrix


# In[14]:


y_pred=cbc.predict(x_test)


# In[15]:


print(classification_report(y_test,y_pred))


# In[16]:


print(confusion_matrix(y_test,y_pred))


# In[17]:


tl=TomekLinks()


# In[18]:


x_tl,y_tl=tl.fit_resample(x,y)


# In[19]:


x_tl.shape


# In[20]:


y_tl.shape


# In[21]:


y_tl.value_counts()


# In[22]:


y.value_counts()


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x_tl,y_tl,random_state=13)


# In[24]:


cbct=CatBoostClassifier(iterations=300,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[25]:


cbct.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[26]:


y_pred=cbct.predict(x_test)


# In[27]:


print(classification_report(y_test,y_pred))


# In[28]:


from imblearn.over_sampling import SMOTE


# In[29]:


sm=SMOTE(random_state=13)


# In[30]:


x_sm,y_sm=sm.fit_resample(x,y)


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x_sm,y_sm,random_state=13)


# In[32]:


y_sm.value_counts()


# In[33]:


cbcs=CatBoostClassifier(iterations=300,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[34]:


cbcs.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[35]:


y_pred=cbcs.predict(x_test)


# In[36]:


print(confusion_matrix(y_test,y_pred))


# In[37]:


print(classification_report(y_test,y_pred))


# In[38]:


from imblearn.combine import SMOTETomek


# In[39]:


smt=SMOTETomek()
x_smt,y_smt=smt.fit_resample(x,y)


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x_smt,y_smt,random_state=13)


# In[41]:


cbcst=CatBoostClassifier(iterations=300,learning_rate=0.1,depth=6,random_state=13,task_type='GPU',verbose=10,eval_metric='Accuracy')


# In[42]:


cbcst.fit(x_train,y_train,cat_col,eval_set=(x_test,y_test))


# In[43]:


y_smt.value_counts()


# In[44]:


y_pred=cbcst.predict(x_test)


# In[45]:


print(classification_report(y_test,y_pred))


# In[46]:


print(confusion_matrix(y_pred,y_test))


# In[ ]:




