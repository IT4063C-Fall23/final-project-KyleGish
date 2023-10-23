#!/usr/bin/env python
# coding: utf-8

# # Water Pollutants Impact on Human Health📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->

# Looking at water pollution and its impact on health and well being. Analyzing different pollutants and health problems could help to provide some insight into how they correlate with one another. Clean water is becoming more scarce each year and more pollutants are impacting water sources. With a lack of regulations and water pollution being commonplace, it will be important to understand the impact of water pollution on health in order to draw sufficient attention and funding to cleanups. Addressing water pollution is critical to saving lives across the globe and this problem will only continue to grow each year as the problem continues.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 

# I want to specifically focus on how the most significant water pollutants impact human health. Being able to break down significant water pollutants and different health data sets could allow for different correlations or patterns to be seen between particular pollutants and the health risks associated with them.
# 
# In combination with focusing on signifant water pollutants and their associated health risks, I would to focus it based on locational data. I will hopefully be able to answer the overall question along with locational risks!

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 

# Water pollutants do have a significant and noticeable impact on human health. Analyzing the water pollutant data as well as health data sets allows us to see a bigger picture and we are able to correlate different pollutants with different health risks. Utilizing the data vizualizations will visually demonstrate the correlaations between different pollutants and possible health risks associated with them. In order to not have such a broad focus, I will be prioritizing the most significant categories of pollutants (top 3-5) and the associated health risks with each. 
# ***
# I hope to be more particular on specific types of pollutants and specific health problems after looking further at the data as well as providing in-depth charts for a correlation. For example, an answer could look like: 
# 
# Chemical spills within water sources cause a significant danger to human life around the area. A noticeble trend of health issues is visualized when a spill pollutes an area. 
# 
# Plastic pollutants across the globe do not have a direct indication on danger to human life, however, more studies and information are demonstrating just how dangerous plastics can be to humans. (more of a general focus on the potential dangers of plastics/microplastics due to how prevelant this particular pollutant is)
# 
# Dangerous diseases can pollute unsanitized waters. These areas of polluted water sources can see spikes of cases of cholera, diarrhoea, dysentery, polio, or hepatitis. (focusing on the spread or dangers of these pollutants on a locational basis could provide some crucial information)
# 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->

# There are several different data sources I will be pulling from for this project. I have yet to figure out which data sources will be most usefull/accurate but as I move through the project, I will iron out which data sources provide the best data that I need for the question: How water pollutants impact human health? I plan to relate these data sets mostly through the location that the tests are done at. This will allow me to correlate a particular pollutant with a trent of water related health risks in an area. There are also particular data sets from the WHO that focus on particular data attributed to human health risks that I can use to verify / as a starting point for other data. 
# 
# Data Source 1: WHO Data (could possibly use other datasets to correlate particular water pollutants in the future)
# 
#     MoralityRate.csv is a csv file that has morality from unsafe water, unsafe sanitation / water hygine
#     DrinkingSanitation.csv is a csv file that notes percentage of water sanitation services per location
#     DiseaseData.csv is a csv file that notes total deaths from cholera per location
# 
# Data Source 2: EPA Water Quality (Hopefully, dataset download has failed multiple times on me as of now but the data looks promising as it takes it data from water stations around the world)
# 
# Data Source 3: IHME
# 
#     death-rates-unsafe-water.csv looks at death rate related from unsafe water globally
# 
# Data Source 4: Kaggle API
# 
#     https://www.kaggle.com/datasets/cityapiio/world-cities-air-quality-and-water-polution/data
# 
# Data Source 5: Data.gov
# 
#     microplastic_pollutants.csv looks at global levels of microplastics and the size of them
# 
# 
# These are the five different sources that I have right now. I hope to expand on some of these data sets as well as utilize other data sets in the future to help ensure that I have accurate information regarding pollutants.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 

# This project is quite complex. There are a lot of different pollutants and data sets that are involved with water quality. Since I will be looking at different water pollutants and health problems, the best way to indicate correlation will be location. The datasets have location tied into them, with WHO datasets listing particular health problems attributed with water pollution. The location will allow me to tie different water related health problems with their location and overall water quality. Hopefully, I will be able to see patterns of higher polluted/contaminated areas with increased health risks in particular areas. Furthermore, I can compare different datasets in order to determine how different pollutants impact different health conditions. I will hopefully be able to create charts to help visualize the data between the different datasets in order to draw meaningful conclusions. 

# In[ ]:


# Start your code here


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[1]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

