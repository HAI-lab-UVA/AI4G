#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df_allocation_history = pd.read_csv("./Data/allocation_history.csv")
df_initial_priority = pd.read_csv("./Data/nutritional_needs.csv")
nutritional_needs = df_initial_priority.iloc[:10,:]
df_allocation_history = df_allocation_history.iloc[:,:20]
days = 7
users = 10
df_priority = df_initial_priority
for i in range(days):
    for j in range(users):
        if i == 0 & j ==0:
            break
        else: 
            unnorm = df_priority.iloc[(i-1)*10+j,1:] + nutritional_needs.iloc[j,1:] - df_allocation_history.iloc[(i-1)*10+j,1:]
            unnorm[unnorm<0]=0
            df_priority.iloc[i*10+j,1:] = unnorm/(np.linalg.norm(unnorm))
for j in range(users):
    df_priority.iloc[j,1:] = df_priority.iloc[j,1:]/(np.linalg.norm(df_priority.iloc[j,1:]))
df_priority.to_csv("./Data/norm_priority.csv",index=False)   


# In[ ]:




