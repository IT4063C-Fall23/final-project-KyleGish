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

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# - https://apps.who.int/gho/data/node.main
# - https://datacatalog.worldbank.org/
# - https://www.healthdata.org/
# - https://ourworldindata.org/burden-of-disease

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

