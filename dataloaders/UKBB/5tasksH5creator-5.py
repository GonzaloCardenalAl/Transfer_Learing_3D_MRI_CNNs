#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os, sys
from glob import glob
from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=False)
import json
from tqdm.notebook import tqdm
from ukbb2020_dataloader import UKBB2020

sys.path.insert(0, "../../helper/")
from plotGraphs import *


# In[30]:


dataset = UKBB2020()


# In[32]:


df = dataset.get_metadata(predefined=[], cols = ["31-0.0", #sex
                                                 '21003-2.0', #age
                                                 '26521-2.0', #total brain volume
                                                 '54-2.0', #assestment center
                                                 "20021-2.0",#SRT estimate right ear
                                                 "1558-2.0",#Alc int freq
                                                 "25061-2.0", #Mean FA in fornix on FA skeleton
                                                ],split='train', rename_cols=False)  


# In[33]:


#Adds an icd label as a column
dficd = dataset.get_metadata(predefined=['icd'], cols=[], print_cols=False, split='all', rename_cols=False)
dfmooddis = dficd.apply(lambda row: row.astype(str).str.contains('F3').any(), axis=1)
df['mood_disorder'] = dfmooddis


# In[34]:


df["mood_disorder"] = df["mood_disorder"].astype(float)


# In[35]:


#We now convert the 6 bins of Alc freq to 3 
df['1558-2.0']= df['1558-2.0'].replace(['1.0','2.0'],'1.0') #Never/ Special -> Rarely
df['1558-2.0']= df['1558-2.0'].replace(['3.0','4.0'],'2.0') #One to 3 month/ one or two week -> ocassional drinkers
df['1558-2.0']= df['1558-2.0'].replace(['5.0','6.0'],'3.0') #3 to 4 week/daily -> frecuent drinkers
df['1558-2.0']= df['1558-2.0'].replace(['-3.0'],np.nan) #Not answered


# In[36]:


#Adding a duplicate column of SRT and dividing it in 3 categories to perform classifcation
df['SRT_right_ear_classification'] = df['20021-2.0']
bins = [-12.0, -7.0, -3.0, np.inf]
categories = ['1.0', '2.0', '3.0'] #good, medium and bad srt threshold
df['SRT_right_ear_classification']= pd.cut(df['SRT_right_ear_classification'], bins, labels = categories)


# In[37]:


df = df.rename(columns={'31-0.0' : 'Sex',
                        '21003-2.0' : 'Age',
                        "20021-2.0" : 'SRT_right_ear',
                        "1558-2.0" :'Alc_int_freq' ,
                        "25061-2.0" : 'Mean_FA_fornix',
                        '26521-2.0': 'Total_brain_volume',
                        '54-2.0': 'Site',
                       })


# In[39]:


dataset.add_var_to_h5(df, 'Sex', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Age', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'SRT_right_ear', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Alc_int_freq', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Mean_FA_fornix', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'mood_disorder', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'SRT_right_ear_classification', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Total_brain_volume', typ='lbl', viz=False)
dataset.add_var_to_h5(df, 'Site', typ='lbl', viz=False)


# In[40]:


#dataset.df_h5 = dataset.df_h5.sample(40682)


# In[41]:


dataset.prepare_X(mri_col='path_T1_MNI')


# In[ ]:


#%%time  ### ! I had a problem here because of the slugfy and file names are not properly set up
dataset.save_h5(filename_prefix="5tasks35k", mri_kwargs={'z_factor':(0.525)})

