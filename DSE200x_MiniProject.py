
# coding: utf-8

# UCSanDiegoX: DSE200x Python for Data Science

# Rustan, 7/5/2018

# # Mini Project: World Development Indicators

# ## Explore renewable energy consumption and its relationship with gross domestic income in the US

# EXISTING VARIABLES:
# Renewable energy consumption (% of total final energy consumption): EG.FEC.RNEW.ZS (1990-2012);
# Gross domestic income (constant 2005 USdollar): NY.GDY.TOTL.KD (1960-2014);
# year variable: Year;
# country variable: CountryCode

# dataset source: https://www.kaggle.com/worldbank/world-development-indicators

# Import libraries: pandas, numpy, matplotlib.pyplot

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read in World Development Indicators dataset

# In[2]:


data = pd.read_csv('.\world-development-indicators\Indicators.csv')


# Check dataset shape

# In[3]:


data.shape


# In[3]:


data.head()


# ## Explore US renewable energy consumption over time

# In[4]:


# Select USA and renewable energy consumption using dynamic values
hist_indicator = 'EG.FEC.RNEW.ZS'
hist_country = 'USA'

maskindicator = data['IndicatorCode'] == hist_indicator 
maskcountry = data['CountryCode'].str.contains(hist_country)

indicator_country = data[maskindicator & maskcountry]


# In[5]:


indicator_country.head()


# In[6]:


indicator_country.tail()


# Plot US renewable energy consumption over time in a bar chart

# In[13]:


# Descriptive statistics for USA renewable energy
indicator_country.describe()


# In[15]:


years = indicator_country['Year'].values
ivalues = indicator_country['Value'].values

plt.bar(years,ivalues)
plt.show()


# Plot US renewable energy consumption over time in a line chart, style

# In[16]:


plt.plot(indicator_country['Year'].values, indicator_country['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel('% of total energy consumption')

# Label the figure
plt.title('Renewable Energy Consumption in USA')

# Set axis max, min
plt.axis([1989, 2012,0,9])

plt.show()


# _This line chart highlights that although the overall trend is for renewable energy consumption to increase, there are significant periods of stagnant or even decreasing renewable energy consumption._

# ## Explore renewable energy consumption by country in 2012

# In[19]:


# Select percentage of renewable energy consumption in 2012 using dynamic values
hist_indicator = 'EG.FEC.RNEW.ZS'
hist_year = 2012

maskindicator = data['IndicatorCode'] == hist_indicator
maskyear = data['Year'] == hist_year

indicator_year = data[maskindicator & maskyear]


# In[21]:


indicator_year.head()


# In[22]:


indicator_year.tail()


# In[23]:


# How many countries have renewable energy consumption data in 2012?
print(len(indicator_year))


# In[24]:


# Plot a histogram of of renewable energy consumption in 2012 by country

# Subplots returns a tuple with the figure, axis attributes
fig, ax = plt.subplots()

ax.annotate("USA, 7.9%",
            xy=(8, 45), xycoords='data',
            xytext=(15, 45), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

plt.hist(indicator_year['Value'], 10, normed=False, facecolor='green')

plt.xlabel('% of total energy consumption')
plt.ylabel('# of Countries')
plt.title('Histogram of Renewable Energy Consumption in 2012')

plt.grid(True)

plt.show()


# _Although the percentage of the total energy consumed that is renewable is relatively low in the US, this histogram makes it clear that similar levels of energy consumption are most common (about 28% of countries fall into the lowest category of approximately 0-8% renewable energy consumption)._

# ## Plot relationship between renewable energy consumption and GDI

# In[26]:


# Select GDI for the United States
hist_indicator2 = 'NY.GDY.TOTL.KD'
hist_country = 'USA'

maskindicator2 = data['IndicatorCode'] == hist_indicator2
maskcountry = data['CountryCode'].str.contains(hist_country)

indicator2_country = data[maskindicator2 & maskcountry]


# In[27]:


indicator2_country.head()


# In[28]:


indicator2_country.tail()


# In[29]:


# Plot USA GDI over time
plt.plot(indicator2_country['Year'].values, indicator2_country['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel('Gross Domestic Income (in trillions)')

# label the figure
plt.title('USA Gross Domestic Income (2005 US$)')

# Set axis ranges
plt.axis([1959, 2010,0,12000000000000])

plt.show()


# _The line plot indicates that the USA GDI increased fairly steadily between 1960 and 2006._

# ## Plot a scatterplot comparing renewable energy consumption and GDI

# In[30]:


# Trim data to 1990-2006 (scatterplot requires equal length arrays)
indicator2_country_trunc = indicator2_country[indicator2_country['Year'] >= 1990]
print(len(indicator2_country_trunc))
indicator_country_trunc = indicator_country[indicator_country['Year'] <= 2006]
print(len(indicator_country_trunc))


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, axis = plt.subplots()

# Set grid lines, Xticks, Xlabel, Ylabel
axis.yaxis.grid(True)
axis.set_title('Renewable Energy Consumption by GDI',fontsize=10)
axis.set_xlabel(indicator2_country_trunc['IndicatorName'].iloc[0],fontsize=10)
axis.set_ylabel('Renewable energy consumption (% total energy)',fontsize=10)

X = indicator2_country_trunc['Value']
Y = indicator_country_trunc['Value']

# Set axis ranges
plt.axis([6000000000000, 12000000000000,0,6.5])

axis.scatter(X, Y)
plt.show()


# _The scatterplot suggests a positive correlation between renewable energy consumption and GDI._

# In[32]:


# Check correlation coefficient
np.corrcoef(indicator2_country_trunc['Value'],indicator_country_trunc['Value'])


# _A .8 correlation is positive and very strong, indicating that in the US, renewable energy consumption increases as GDI increases._
