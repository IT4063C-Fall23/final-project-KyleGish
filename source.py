#!/usr/bin/env python
# coding: utf-8

# # Income and Wealth's Impact on Communicable Diseases📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->

# Looking at the spread of diseases is crucial to helping improve human health and well beings. Breaking down the Income and Disease numbers could help to provide some insight into how they correlate with one another. Diseases can become a very big problem very quickly. We got to see this play out during the chaos when Covid-19 was spreading around. Analyzing the impact between income and infections could hopefully draw more attention to situations like what happened with Covid-19. Addressing the spread of diseases is critical to saving lives across the globe and this problem will only continue to grow as diseases adapt and evolve overtime.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 

# I want to specifically focus on how significant diseases impact human health in different areas of income / around the globe. Being able to break down significant diseases and income datasets could allow for different correlations or patterns to be seen between the impact of these diseases and the income area they are associated in. Would we be able to draw potential conclusions from this data to help save lives or better address these diseases in different income areas? Focusig the data around locational data will provide me with some targeted data that I can break down to better answer the main question. 
# 

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 

# 
# Income does have a significant and noticeble effect on how impactful diseases are to human health. Analyzing the disease datasets as well as the income data sets allows us to see the bigger picture as we able able to correlated different different income with different disease risks. Utilizing the data vizualizations with demonstrate the correlations between income and disease impact. In order to help limit the focus, I will be prioritizing the most significant and impactful diseases.
# 
# 
# ***
# I hope to be more particular and provide clearer correlations after breaking down the data. For example, an answer could look like: 
# 
# Cholera has a significant impact on lower income communities. Noticeble trends between lower income areas and higher infections / deaths from cholera is noted.
# 
# Higher income areas report higher level of vaccinations which decrease disease impact on higher income locations. The reverse is also seen with lower income areas reporting lower level of vaccinations and higher levels of infections and deaths. 
# 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->

# There are several different data sources I will be pulling from for this project.
# 
# 
# Data Source 1: WHO Data (Wide Range of Communicable Diseases Datasets - temporary)
# 
#     https://apps.who.int/gho/data/node.main
#     
#     The CSV's for WHO disease data are located in the OtherDisease folder in the Data folder. The primary focus will be cholera with some analytics towards other Diseases as well. 
# 
# Data Source 2: Global Income Data
# 
#     https://datacatalog.worldbank.org/
# 
#     income_category.csv is a file that notes categorical income levels based off country
# 
# Data Source 3: IHME Global Burden of Disease (2019) Study
# 
#     https://www.healthdata.org/
# 
#     gbd_countries.csv is a dataset that includes country, year, type of disease, and number of deaths / DALYs
# 
# Data Source 4: Our World In Data
# 
#     https://ourworldindata.org/burden-of-disease
# 
#     gbd_communicable_diseases.csv is a refined dataset that includes the country, country code, year, and DALYs values
# 
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 

# This project is quite complex. I want to break down different communicable diseases and their impact on lower income countries. There are a lot of different ways to break this down so I wanted to focus on looking at the overall income of different countries and then comparing that to different disease cases, total deaths, and DALYs values to see how they impact different income communities. 
# 
# I wanted to focus on a few diseases such as cholera and measles from the WHO datasets first to see if I can establish a patter before working with the bigger sets of data. I then want to move on to the GBD data to get a better picture of the overall data and its impact. I want to analyze how the different statistics differ based on location and income to help correlate the data. 
# 
# I plan on doing this by almost breaking down the analysis and visualizations into different sections to help draw conclusions. I want to draw meaningful conclusions so I will do my best to provide accurate and meaningful visualizations. 

# ### Checkpoint 1 Overview
# 
# - I was able to import the different datasets that I wanted to work with and was able to give a summary of my plans for the next step. 

# ## Exploratory Data Analysis and Visualization

# In[813]:


import sys
assert sys.version_info >= (3, 10)

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from scipy.stats import trim_mean
import os


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
  OneHotEncoder,
  OrdinalEncoder,
  StandardScaler
)
from sklearn.impute import (
  SimpleImputer
)

from sklearn.model_selection import (
  StratifiedShuffleSplit,
  train_test_split,
  cross_val_score,
  KFold,
  GridSearchCV
)

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
  RandomForestClassifier, 
  GradientBoostingClassifier,
  BaggingClassifier
)

import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")


# In[814]:


income_cat = pd.read_csv('Data/Income/income_category.csv')


# In[815]:


cholera_cases = pd.read_csv('Data/Cholera/cholera1_data.csv')
cholera_deaths = pd.read_csv('Data/Cholera/cholera2_data.csv')


# In[816]:


measles_data = pd.read_csv('Data/OtherDiseases/measles_data.csv')


# In[817]:


gbd_data_countries = pd.read_csv('Data/GBD/gbd_countries.csv')
income_grouping_data = pd.read_csv('Data/GBD/revised_world_bank_data.csv')
gbd_cd_data = pd.read_csv('Data/GBD/gbd_communicable_diseases.csv')


# ## Exploratory Data Analysis (EDA)

# My goal is to breakdown how income correlates with communicable disease's impact one different countries. I first want to take a look at the income datasets to set a base to go off of.

# ### Income Information

# In[819]:


display(income_cat.head(5))


# In[820]:


display(income_cat.shape)


# In[821]:


display(income_cat.info())


# In[822]:


display(income_cat.isnull().sum())


# In[823]:


display(income_cat.describe())


# Since there are no huge outliers, duplicate values, or anomalies in the data we don't have to focus too much on cleaning the data. However, there is one missing value for the income group for income_category.csv. So I will focus on cleaning up that value.

# In[824]:


income_cat_missing = income_cat[income_cat.isnull().any(axis=1)]
print(income_cat_missing)


# The data comes with a note that lists the breakdown: 
# 
# This table classifies all World Bank member countries (189), and all other economies with populations of more than 30,000. For operational and analytical purposes, economies are divided among income groups according to 2022 gross national income (GNI) per capita, calculated using the World Bank Atlas method. The groups are: low income, $1,135 or less; lower middle income, $1,136  to $4,465; upper middle income, $4,466 to $13,845; and high income, $13,846 or more. The effective operational cutoff for IDA eligibility is $1,315 or less.
# 
# https://data.worldbank.org/indicator/NY.GNP.PCAP.CD?locations=VE
# 
# Venezuela's most recent value is listed at 13,010 putting in the upper middle income. 

# In[825]:


missing_value = income_cat_missing.index[0]

income_cat.loc[missing_value, 'IncomeGroup'] = 'Upper middle income'


# In[826]:


income_cat.isnull().sum()


# In[827]:


display(income_cat.value_counts('IncomeGroup'))


# In[828]:


display(income_cat['IncomeGroup'].value_counts().plot(kind='bar', xlabel='IncomeGroup', ylabel='Count', rot=-45))


# This graph includes general information of how many of each different income group there are

# In[829]:


# Loaded country boundaries - (GeoJSON file)
country_boundries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

merged_income_cat = country_boundries.merge(income_cat, how='left', left_on='iso_a3', right_on='Code')

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_income_cat.plot(column='IncomeGroup', cmap='viridis', ax=ax, legend=True)
plt.title('Income Group by Country')
plt.show()


# This is a geomap of the income grouping of each country. This project will explore how income impacts human life. Hopefully we will begin to see patterns emerge. Maybe countries indicated as low or lower middle income having a correlation with greater negative impact to human life. As we explore the diseases by country, we will be able to relate back to the income information throughout the project.

# ### Cholera Information

# The two cholera datasets include different values for cases, deaths, and fatality rates. Instead of having them in two different datasets, I want to combine them. 

# In[831]:


display(cholera_cases.head(5))
display(cholera_deaths.head(5))


# In[832]:


cholera_data = pd.merge(cholera_cases, cholera_deaths, how='left', on=['Countries, territories and areas','Year'])

cholera_data.head()


# In[833]:


cholera_data.shape


# In[834]:


cholera_data.isnull().sum()


# In[835]:


cholera_data.info()


# In[836]:


cholera_data.describe()


# In[837]:


#To convert the object types to int64 after merging
cholera_data['Number of reported deaths from cholera'].replace('Unknown', np.nan, inplace=True)
cholera_data['Number of reported deaths from cholera'] = cholera_data['Number of reported deaths from cholera'].astype(float).astype('Int64')


# In[838]:


cholera_data['Cholera case fatality rate'] = cholera_data['Number of reported deaths from cholera'] / cholera_data['Number of reported cases of cholera']


# In[839]:


cholera_data.info()


# In[840]:


cholera_data.head()


# In[841]:


cholera_data.describe()


# In[842]:


cholera_data.isnull().sum()


# In[843]:


cholera_data_missing = cholera_data[cholera_data.isnull().any(axis=1)]
print(cholera_data_missing)


# Since much of the missing data is just missing records from different years, I believe the best case would be to drop the rows that are not consistent across the data to help provide a better picture of the data. Most of the data is still available and I do not believe that dropping the rows will skew the data especially since we could look at years where all of the data is present across the graphs. 

# In[844]:


cholera_data_prepared = cholera_data.dropna()


# In[845]:


cholera_data_prepared.isnull().sum()


# In[846]:


cholera_data_aggregated = cholera_data_prepared.groupby('Countries, territories and areas').agg({'Number of reported cases of cholera': 'mean','Number of reported deaths from cholera': 'mean'}).reset_index()
cholera_data_filtered = cholera_data_aggregated[(cholera_data_aggregated['Number of reported cases of cholera'] != 0) & (cholera_data_aggregated['Number of reported deaths from cholera'] != 0)]

plt.figure(figsize=(24, 12))


sns.barplot(x='Countries, territories and areas', y='Number of reported cases of cholera', data=cholera_data_filtered, label='Reported Cases', color='cyan')
sns.barplot(x='Countries, territories and areas', y='Number of reported deaths from cholera', data=cholera_data_filtered, label='Reported Deaths', color='red')
plt.xticks(rotation=90)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.title('Total Reported Cases vs Deaths by Country')
plt.legend()
plt.tight_layout()
plt.show()


# In[847]:


cholera_fatality_rate = cholera_data_prepared.groupby('Countries, territories and areas')['Cholera case fatality rate'].mean().reset_index()

plt.figure(figsize=(24, 12))
sns.barplot(x='Countries, territories and areas', y='Cholera case fatality rate', data=cholera_fatality_rate)
plt.xticks(rotation=90)
plt.xlabel('Countries')
plt.ylabel('Cholera Case Fatality Rate')
plt.title('Average Cholera Case Fatality Rate by Country')
plt.tight_layout()
plt.show()


# These two graphs, while a bit crowded, do provide a lot of information. We see the two big outliers in graph one with Haiti and Peru having a large number of cases of Cholera. However, we do not see that translate to fatality rate. The second graph has Italy and Oman as the two outliers but they were not outliers on the first graph. It seems some of the data is skewed which may indicate some of the rates are off.

# In[848]:


cholera_rate_outliers = cholera_data_prepared[cholera_data_prepared['Cholera case fatality rate'] > 1.0]
print(cholera_rate_outliers)


# In[849]:


cholera_data_prepared.drop(index=1093, inplace=True)


# In[850]:


cholera_rate_outliers = cholera_data_prepared[cholera_data_prepared['Cholera case fatality rate'] > 1.0]
print(cholera_rate_outliers)


# In[851]:


cholera_fatality_rate = cholera_data_prepared.groupby('Countries, territories and areas')['Cholera case fatality rate'].mean().reset_index()

plt.figure(figsize=(24, 12))
sns.barplot(x='Countries, territories and areas', y='Cholera case fatality rate', data=cholera_fatality_rate)
plt.xticks(rotation=90)
plt.xlabel('Countries')
plt.ylabel('Cholera Case Fatality Rate')
plt.title('Average Cholera Case Fatality Rate by Country')
plt.tight_layout()
plt.show()


# Italy is no longer visualized on the graph

# In[852]:


cholera_rate_sort = cholera_fatality_rate.sort_values(by='Cholera case fatality rate', ascending=False)
cholera_highest_fatality_rate = cholera_rate_sort.nlargest(10, 'Cholera case fatality rate')

plt.figure(figsize=(24, 12))
sns.barplot(x='Countries, territories and areas', y='Cholera case fatality rate', data=cholera_highest_fatality_rate)
plt.xticks(rotation=90)
plt.xlabel('Countries')
plt.ylabel('Cholera Case Fatality Rate')
plt.title('Highest Fatality Rates')
plt.tight_layout()
plt.show()


# I wanted to see the top countries with the highest fatality rates and then will see how they line up with the income groupings.

# In[853]:


countries_check = ['Oman', 'Bangladesh', 'Czechia', 'Myanmar', 'Cambodia', 'India', 'Mali', 'Djibouti', 'Zambia', 'Lao PDR']

for country in countries_check:
    country_rate_check = income_cat[income_cat['Economy'] == country][['Economy', 'IncomeGroup']]
    print(country_rate_check)


# In[854]:


cholera_highest_fatality_rate = cholera_rate_sort.nlargest(62, 'Cholera case fatality rate')

highest_rates = cholera_highest_fatality_rate['Countries, territories and areas'].tolist()

filtered_countries = income_cat[income_cat['Economy'].isin(highest_rates)][['Economy', 'IncomeGroup']]

income_group_count = filtered_countries['IncomeGroup'].value_counts()

print(income_group_count)


# In[855]:


#Overall Income Categories
display(income_cat['IncomeGroup'].value_counts().plot(kind='bar', xlabel='IncomeGroup', ylabel='Count', rot=-45))


# This is the total count of income groups included in the income_cat dataset

# In[856]:


display(income_group_count.plot(kind='bar', xlabel='IncomeGroup', ylabel='Count', rot=-45))


# There are some missing values due to the differences in the dataset with some countries not being listed between the two datasets. However, we can see from the original IncomeGroup graph compared to the one revised based off of the fatality rates that there is a significance that the fatality rate plays on the income grouping. 
# 
# The original income group has 63.3% in the Upper middle income and High income categories.
# The top 50 country matches between fatality rate and income_cat has only 34% in the Upper middle income and High income categories. 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# - https://apps.who.int/gho/data/node.main
# - https://datacatalog.worldbank.org/
# - https://www.healthdata.org/
# - https://ourworldindata.org/burden-of-disease

# In[830]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

