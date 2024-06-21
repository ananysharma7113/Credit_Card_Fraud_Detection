#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score


# In[2]:


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# In[3]:


df=pd.read_csv('creditcard.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


fraud_check = pd.value_counts(df['Class'], sort=True)
fraud_check.plot(kind='bar',rot=0,color='r')
plt.title("Normal and Fraud Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
labels=['Normal','Fraud']
plt.xticks(range(2),labels)
plt.show()


# In[7]:


fraud_people = df[df['Class']==1]
normal_people = df[df['Class']==0]


# In[8]:


fraud_people.shape


# In[9]:


normal_people.shape


# In[10]:


df.corr()
plt.figure(figsize=(30,30))
g=sns.heatmap(df.corr(),annot=True)


# In[11]:


columns=df.columns.tolist()
columns = [var for var in columns if var not in ["Class"]]
target="Class"
x=df[columns]
y=df[target]


# In[12]:


x.shape


# In[13]:


y.shape


# In[14]:


x.head()


# In[15]:


y.head()


# In[16]:


from sklearn.model_selection import  train_test_split


# In[17]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30,random_state=42)


# # ML

# In[18]:


iso_forest = IsolationForest(n_estimators=100, max_samples=len(x_train),verbose=0)


# In[19]:


iso_forest.fit(x_train,y_train)


# In[20]:


ypred=iso_forest.predict(x_test)
ypred


# In[21]:


ypred[ypred == 1]=0
ypred[ypred == -1]=1


# In[22]:


print(accuracy_score(y_test,ypred))
print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,ypred)


# In[ ]:




