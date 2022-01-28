#!/usr/bin/env python
# coding: utf-8

# # Modeling HW - Cam Pitcher

# ## Import Packages and Data

# In[236]:


## import packages

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics

## import data 
data = pd.read_csv('./marketing_modified.csv')
# # Exploratory Data Analysis
# ## Check for missing values
#make a copy of the base data to modify
modified_data = data.copy(deep = True)
# #### Drop the "Unnamed: 0" column (row number)
modified_data = modified_data.drop(columns=('Unnamed: 0'))
# #### What's missing??
# It's a good thing that there aren't any missing IDs.
# ### Check type of variables
# #### Check for duplicate IDs
# Since there aren't any duplicate IDs, we can remove this column
modified_data = modified_data.drop(columns=('ID'))
# ## Check out the object datatypes
# ### Check the unique values of our object datatypes (make sure they should be objects)
#grab all objects
objects = modified_data.select_dtypes(include = ['object']).columns.to_list()
# ### Fix Marital Status
# - YOLO isn't a real/valid marital status, and alone can be combined with Single. 
# - Together is a little ambiguous, but since it has plenty of entries I won't worry about it
modified_data['Marital_Status'] = [float('NaN') if status == 'YOLO' else status for status in modified_data.Marital_Status]
modified_data['Marital_Status'] = ['Single' if status == 'Alone' else status for status in modified_data.Marital_Status]
# ### Check Factor Levels
# There are not balanced levels in our factors, most notably:
# 
# - There are ~12x more Married than Widow
# - There are ~20x more Graduation than Basic
# ## Check the spread/distribution of our numerical datatypes
numerical_features = modified_data.select_dtypes(include = ['float64', 'int64']).columns.to_list()
# ### See if any of the numerical columns are actually categorical features

#print column value counts for lists with 10 or fewer elements
# ##### Z_CostContact isn't showing up for some reason
modified_data['Z_CostContact'].value_counts()
# ##### All values in Z_CostContact are the same. Let's drop it
modified_data = modified_data.drop(columns=('Z_CostContact'))
numerical_features.remove('Z_CostContact')
# #### Can we combine AcceptedCmp1-5 into one column?

# Can you only have a positive response in one of these columns?
accepted_cmp = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
# It looks like there isn't exclusivity between columns. Since all 5 columns are binary, instead we'll make them into yes/no object columns.
# 
# Z-Revenue has nothing
modified_data = modified_data.drop(columns=('Z_Revenue'))
numerical_features.remove('Z_Revenue')
# ##### Response, Complain and the AcceptedCmp1-5 are all Binary (yes/no). We need to change those to be categorical.
toCategorical = ['Response', 'Complain']
toCategorical.extend(accepted_cmp)

for c in modified_data[toCategorical].columns:
    modified_data[c] = ['Yes' if num == 1.0 else 'No' if num == 0.0 else None for num in modified_data[c]]

    modified_data[c].unique()

    objects.append(c)
    toCategorical.remove(c)
    numerical_features.remove(c)

# It shoud be noted that the factor levels are very unbalanced for all 7 of the variables we just reclassified. Complaints specifically is concerning with only 21 complaints filed. We'll leave this for now
# Most of these look fine to me, I am not sure if kidhome/dependenthome/teenhome need to be put as categorical but since they do scale numerically I'm leaving them in for now
# Looking good, now we just need to take care of the missing values in Recency, Education, Marital Status, and Income.
# ## Missing Values
# For Education and Marital Status, since they are categorical, I'll create an "Undisclosed" category since I don't feel comfortable imputing categorical data here.
# ### Replacing NaN with "Unknown" in Education and Marital Status
modified_data[['Education', 'Marital_Status']] = modified_data[['Education', 'Marital_Status']].fillna('Unknown')

modified_data['Income'].fillna(value=modified_data['Income'].mean(), inplace=True)
modified_data['Recency'].fillna(value=modified_data['Recency'].mean(), inplace=True)

# #### Format the dataset and encode
# I'm going to drop Education, Marital Status, and Dt_Customer to keep the model and encoding simpler

replace_dict = {"Yes" : 1, "No" : 0}
encoded_data = modified_data.drop(['Education','Marital_Status','Dt_Customer'], axis = 1).replace(replace_dict)

#set response variable
target = encoded_data.Response
X = encoded_data.drop(['Response'], axis = 1)

# Get 20% test and 80% train
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=134)

