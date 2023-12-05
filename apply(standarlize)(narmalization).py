#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd



# In[2]:


df=pd.read_csv("D:\\pandas.excel\\covid_toy.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[11]:


df.info


# In[13]:


df.isnull().sum()


# In[19]:


df['fever'].fillna(df['fever'].mean(),inplace=True)


# In[20]:


df.isnull().sum()


# In[24]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[26]:


for column in df.select_dtypes(include=['object']).columns:
    df[column] = lb.fit_transform(df[column])


# In[27]:


df.head()


# In[28]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[29]:


X = df.drop(['has_covid'], axis=1)
y=df['has_covid']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=42)


# In[33]:


X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.fit(X_test)


# In[37]:


X_train_new=pd.DataFrame(X_train_scaled,columns=X_train.columns)


# In[38]:


np.round(X_train_new.describe(),1)


# # APlly standarliaztion on Attrition data

# In[39]:


import numpy as np
import pandas as pd


# In[41]:


df=pd.read_csv("D:\\pandas.excel\\Attrition.csv")


# In[42]:


df.head()


# In[43]:


df.head()


# In[44]:


df.info()


# In[46]:


df.isnull().sum()


# In[47]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[48]:


for column in df.select_dtypes(include=['object']).columns:
    df[column] = lb.fit_transform(df[column])


# In[49]:


df.head()


# In[50]:


X = df.drop([''], axis=1)
y=df['Attrition']


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=42)


# In[52]:


X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.fit(X_test)


# In[53]:


X_train_new=pd.DataFrame(X_train_scaled,columns=X_train.columns)


# In[54]:


np.round(X_train_new.describe(),1)


# # apply normalization
# 

# import numpy as np
# import pandas as pd

# In[118]:


import numpy as np 
import pandas as pd


# In[119]:


df=pd.read_csv("D:\\pandas.excel\\click.csv")


# In[124]:


df


# In[122]:


df.info()


# In[131]:


# Assuming your DataFrame is named df
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create new columns for year, month, week, and time
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['day'] = df['Timestamp'].dt.day
df['Time'] = df['Timestamp'].dt.time

# Display the updated DataFrame
df.head()


# In[123]:


df.isnull().sum()


# In[77]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[78]:


for column in df.select_dtypes(include=['object']).columns:
    df[column] = lb.fit_transform(df[column])


# In[79]:


df.head()


# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


x=df.drop(columns=['Clicked on Ad'],axis=1)
y=df['Clicked on Ad']


# In[82]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[83]:


from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()


# In[85]:


x_train_mn=mn.fit_transform(x_train)


# In[86]:


x_train_new=pd.DataFrame(x_train_mn,columns=x_train.columns)


# In[87]:


x_train_new


# In[88]:


np.round(x_train.describe(),1)


# # Aplly normalization on indian cities 

# In[104]:


import numpy as np
import pandas as pd


# In[105]:


df=pd.read_csv("D:\\pandas.excel\\Indian_cities.csv")


# In[106]:


df.head()


# In[107]:


df.isnull().sum()


# In[108]:


df.info()


# In[109]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[110]:


for column in df.select_dtypes(include=['object']).columns:
    df[column] = lb.fit_transform(df[column])


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


x=df.drop(columns=['total_graduates'],axis=1)
y=df['total_graduates']


# In[113]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[114]:


from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()


# In[115]:


x_train_mn=mn.fit_transform(x_train)


# In[116]:


x_train_new=pd.DataFrame(x_train_mn,columns=x_train.columns)


# In[117]:


x_train_new


# In[ ]:




