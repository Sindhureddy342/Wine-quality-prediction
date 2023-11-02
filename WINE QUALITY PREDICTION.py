#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[39]:


data = pd.read_csv('/Users/kadiresindhureddy/Downloads/WineQT.csv')


# In[40]:


print("Sample data:")
print(data.head())


# In[41]:


X = data.drop(['quality', 'Id'], axis=1)
y = data['quality']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[44]:


y_pred = model.predict(X_test)


# In[45]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[46]:


print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2): {r2}")


# In[47]:


output = pd.DataFrame({'Actual Quality': y_test, 'Predicted Quality': y_pred})
print("\nActual vs. Predicted Quality of Wine:")
print(output)


# In[48]:


sns.set(style="whitegrid")


# In[49]:


plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs. Predicted House Prices")

# Display a density plot for the points
sns.kdeplot(y_test, shade=True, label='Actual Prices', color='blue')
sns.kdeplot(y_pred, shade=True, label='Predicted Prices', color='red')
plt.legend()
plt.show()

