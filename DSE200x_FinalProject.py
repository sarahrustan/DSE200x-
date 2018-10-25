
# coding: utf-8

# 
# UCSanDiegoX: DSE200x Python for Data Science

# Rustan, 9/4/2018

# # Final Project: Exploring attitudes towards opioids using k-modes clustering algorithm

# ### Objective

# To identify clusters of demographic and experiential variables that are associated with attitudes towards opioids.

# ### Method

# Combine data from mutiple sources to create a variable identifying each survey response geography as urban, suburban, or rural. Conduct cluster analysis using k-modes clustering algorithm (an unsupervised machine learning algorithm for categorical variables) to determine whether there are clusters of demographic and experiential variables associated with attitudes towards opioids, including urban, suburban or rural geography.

# ### Datasources

# 1. Proprietary survey, n = 1,200
#     
#     - Geographic: _VGeoRegion_ (String): State; _Vpostal_ (String): Zip code
#     - Demographic/Experiential: var9, var11, var12, var13, educrec, incomerec, var16, politrec, var217
#     - Attitudinal:  var230, var231, var232, var233, var234, var235, var236, var237rec, var238rec, var239rec, var240rec, var241rec, var242rec, var243rec, var244rec, var245rec, var246, var247rec var248rec, var249rec, var250rec, var251rec, var252rec, var253rec, var254rec, var255rec
#     
# 
# 2. 2015 ZIP Code Tabulation Areas Gazetteer File, n = 33,144 
# https://www.census.gov/geo/maps-data/data/gazetteer2015.html
# 
#     - _GEOID_: Five digit ZIP Code Tabulation Area Census Code; *ALAND_SQMI*: Land Area (square miles)
#     
# 
# 3. B25001 HOUSING UNITS by ZCTA, 2012-2016 American Community Survey 5-Year Estimates, n = 33,120 https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_5YR_B25001&prodType=table
# 
#     - _GEO.id2_: Five digit ZIP Code Tabulation Area Census Code; *HD01_VD01*: Housing Units, Estimated

# ### Results

# Three distinct groups of survey respondents were identified. The first two groups embody attitudes typical of plaintiff- and defense-oriented jurors in similar cases, while the third group appears to have demonstrated a significant acquiescence bias, which is a tendency to either answer 'Yes' or choose the first response option given. 

# As shown below, the graph representing the costs associated with different number of clusters indicates that the 'elbow' occurs at three clusters; in other words, the amount that model fit improves with the addition of each additional cluster decreases after three clusters. This suggests that three clusters is the ideal number for this analysis. However, since identifying differences in attitudes between sub-groups of plaintiff- or defense-oriented jurors  could prove useful in framing the case, future research could include using the k-modes clustering algorithm for 4 (and possibly more) clusters.
# 
# 
# 

# 
# ### Preprocessing and analysis

# #### Identify each Zip Code Tabulation Area as Urban, Suburban, or Rural based on housing density

# ZCTA housing density = Number of housing units in ZCTA/Square miles in ZCTA

# Values:
# 0. Urban: >2,213 households per square mile; 
# 1. Suburban: 102 to 2,213 households per square mile; 
# 2. Rural: <102 households per square mile
# 
# Source: https://fivethirtyeight.com/features/how-suburban-are-big-american-cities/

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


# Read in Opioids survey dataset
df_opioids = pd.read_csv('OpioidsWKNG.csv')


# In[4]:


df_opioids.head()


# In[5]:


# Check zip code frequencies
df_opioids.Vpostal.value_counts()


# In[6]:


# Display rows with one or more null values
df_opioids[df_opioids.isnull().any(axis=1)]


# In[7]:


# Replace missing values ' ' in Vpostal with missing values np.nan objects
df_opioids['Vpostal'].replace(' ', np.nan, inplace=True)


# In[8]:


# Display rows with one or more null values
df_opioids[df_opioids.isnull().any(axis=1)]


# In[9]:


# Drop rows with missing values in Vpostal
df2_opioids=df_opioids.dropna(subset=['Vpostal'])


# In[10]:


# Check number of rows before and after dropna
before_rows = df_opioids.shape[0]
print(before_rows)


# In[11]:


after_rows = df2_opioids.shape[0]
print(after_rows)


# In[12]:


# Check how many of the remaining records have states attached
df2_opioids.VGeoRegion.value_counts()


# In[13]:


df2_opioids['Vpostal']


# In[14]:


# Check how many unique zip codes remain
df2_opioids['Vpostal'].nunique()


# In[15]:


# Create new GEOID variable by transforming Vpostal into an integer (stripping the leading zeros from Vpostal)
strip_Vpostal = df2_opioids['Vpostal'].apply(int)


# In[16]:


strip_Vpostal.head()


# In[17]:


len(strip_Vpostal)


# In[18]:


# Check zip codes count
strip_Vpostal.value_counts()


# In[19]:


df3_opioids = df2_opioids.assign(GEOID = strip_Vpostal)


# In[20]:


df3_opioids.head()


# In[21]:


# Check new column GEOID
GEOIDslice = ['Vrid', 'Vpostal','GEOID']
GEOIDcheck = df3_opioids[GEOIDslice]
print(GEOIDcheck)


# In[22]:


# Save records with zip codes as a new csv file
df3_opioids.to_csv("Opioids_ZCSubset.csv", index=False)


# In[23]:


# Read in ZIP Code Tabulation Areas Gazetteer File
df_ZCTA = pd.read_csv("2015_Gaz_zcta_national.txt", sep = "\t")
df_ZCTA.head()


# In[24]:


len(df_ZCTA.index)


# In[25]:


# Create dataset with GEOID and ALAND_SQMI only, save as .csv
features = ['GEOID', 'ALAND_SQMI']
df2_ZCTA = df_ZCTA[features]


# In[26]:


df2_ZCTA.head()


# In[27]:


# Save land area by ZCTA as a new csv file
df2_ZCTA.to_csv("LandArea_ZCTA.csv", index=False)


# In[28]:


# Read in housing unit counts by ZCTA from ACS data
df_HU = pd.read_csv("ACS_16_5YR_B25001_with_ann.csv", header=0, skiprows=[1])
df_HU.head()


# In[29]:


# Rename GEO.id2 to GEOID
df_HU.rename(columns = {'GEO.id2':'GEOID'}, inplace=True)
df_HU.head()


# In[30]:


# Create dataset with GEOID and HD01_VD01 only, save as .csv
features2 = ['GEOID', 'HD01_VD01']
df2_HU= df_HU[features2]


# In[31]:


df2_HU.head()


# In[32]:


len(df2_HU)


# In[33]:


# Save housing units by ZCTA as a new csv file
df2_HU.to_csv("HousingUnits_ZCTA.csv", index=False)


# In[34]:


# Add housing units variable 'HD01_VD01' from df2_HU to survey dataframe df3_opioids by merging on GEOID
df_merged = pd.merge(df3_opioids, df2_HU, how='left', on='GEOID')
df_merged.head()


# In[35]:


# Add land area variable 'ALAND_SQMI' from df2_ZCTA to merged dataframe df_merged by merging on GEOID
df_merged = pd.merge(df_merged, df2_ZCTA, how='left', on='GEOID')
df_merged.head()


# In[36]:


# Create new variable 'HD' housing density (housing units per square mile = HD01_VD01/ALAND_SQMI)
df_merged['HD'] = df_merged['HD01_VD01']/df_merged['ALAND_SQMI']
df_merged.head()


# In[37]:


# Check number of values in HD
len(df_merged['HD'])


# In[38]:


# Display rows with one or more null values
df_merged[df_merged.isnull().any(axis=1)]


# In[39]:


before_rows = df_merged.shape[0]
print(before_rows)


# In[40]:


# Drop rows with missing values
df_merged = df_merged.dropna()


# In[41]:


after_rows = df_merged.shape[0]
print(after_rows)


# In[42]:


# Check number of rows dropped
before_rows - after_rows


# In[43]:


# Check how many unique zip codes remain
df_merged['GEOID'].nunique()


# In[44]:


# Create new variable 'CT' community type (Urban: >2,213 hu/sqmi; Suburban: 102 to 2,213 hu/sqmi; Rural: <102 hu/sqmi)
df_merged['CT']=np.where(df_merged['HD']>2213, 0,(np.where(df_merged['HD']<102,2,1)))
df_merged.head()


# In[45]:


# Check CT frequencies
df_merged.CT.value_counts()


# In[46]:


# Save merged dataframe as a new csv file
df_merged.to_csv("OpioidsMerged.csv", index=False)


# ### Use k-modes clustering algorithm to identify associated clusters of demographic/experiential variables and opioid attitudes

# In[3]:


from kmodes.kmodes import KModes
import python_utils
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


# Recode var9 into standard range
df_merged['var9r'] = np.where(df_merged['var9']==10001, 1, 
                       (np.where(df_merged['var9']==10002,2,
                                 (np.where(df_merged['var9']==10003,3,
                                          (np.where(df_merged['var9']==10004,4,
                                                   (np.where(df_merged['var9']==10005,5,
                                                             (np.where(df_merged['var9']==10006,6,
                                                                      (np.where(df_merged['var9']==10007,7,
                                                                               (np.where(df_merged['var9']==10008,8,
                                                                                        (np.where(df_merged['var9']==10009,9,10)))))))))))))))))


# In[44]:


# Check recodes
recodes = ['var9', 'var9r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[45]:


# Recode var11 into range (for graphing)
df_merged['var11r'] = np.where(df_merged['var11']==10023, 0, 1 )


# In[46]:


# Check recodes
recodes = ['var11', 'var11r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[48]:


# Recode var12 into range (for graphing)
df_merged['var12r'] = np.where(df_merged['var12']==10025, 1, 
                              (np.where(df_merged['var12']==10026,2,
                                       (np.where(df_merged['var12']==10027,3,
                                                (np.where(df_merged['var12']==10028,4,
                                                         (np.where(df_merged['var12']==10029,5,6)))))))))


# In[49]:


# Check recodes
recodes = ['var12', 'var12r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[50]:


# Recode var13 into range (for graphing)
df_merged['var13r'] = np.where(df_merged['var13']==10031, 1, 
                              (np.where(df_merged['var13']==10032,2,
                                       (np.where(df_merged['var13']==10033,3,
                                                (np.where(df_merged['var13']==10034,4,
                                                         (np.where(df_merged['var13']==10035,5,6)))))))))


# In[51]:


# Check recodes
recodes = ['var13', 'var13r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[53]:


# Recode var16 into range (for graphing)
df_merged['var16r'] = np.where(df_merged['var16']==10052, 1, 
                              (np.where(df_merged['var16']==10053,2,
                                       (np.where(df_merged['var16']==10054,3,
                                                (np.where(df_merged['var16']==10055,4,
                                                         (np.where(df_merged['var16']==10056,5,
                                                                  (np.where(df_merged['var16']==10057,6,7)))))))))))


# In[54]:


# Check recodes
recodes = ['var16', 'var16r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[55]:


# Recode var217 into range (for graphing)
df_merged['var217r'] = np.where(df_merged['var217']==10656, 1, 
                              (np.where(df_merged['var217']==10657,2,
                                       (np.where(df_merged['var217']==10658,3,
                                                (np.where(df_merged['var217']==10659,4,
                                                         (np.where(df_merged['var217']==10660,5,6)))))))))


# In[56]:


# Check recodes
recodes = ['var217', 'var217r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[57]:


# Recode var230 into range (for graphing)
df_merged['var230r'] = np.where(df_merged['var230']==10722, 1, 
                              (np.where(df_merged['var230']==10723,2,
                                       (np.where(df_merged['var230']==10724,3, 4)))))


# In[58]:


# Check recodes
recodes = ['var230', 'var230r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[59]:


# Recode var231 into range (for graphing)
df_merged['var231r'] = np.where(df_merged['var231']==10726, 1, 
                              (np.where(df_merged['var231']==10727,2,3)))


# In[60]:


# Check recodes
recodes = ['var231', 'var231r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[61]:


# Recode var232 into range (for graphing)
df_merged['var232r'] = np.where(df_merged['var232']==10729, 1, 
                              (np.where(df_merged['var232']==10730,2,
                                       (np.where(df_merged['var232']==10731,3,4)))))


# In[62]:


# Check recodes
recodes = ['var232', 'var232r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[63]:


# Recode var233 into range (for graphing)
df_merged['var233r'] = np.where(df_merged['var233']==10733, 1, 
                              (np.where(df_merged['var233']==10734,2,
                                       (np.where(df_merged['var233']==10735,3,4)))))


# In[64]:


# Check recodes
recodes = ['var233', 'var233r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[65]:


# Recode var234 into range (for graphing)
df_merged['var234r'] = np.where(df_merged['var234']==10737, 1, 
                              (np.where(df_merged['var234']==10738,2,
                                       (np.where(df_merged['var234']==10739,3,4)))))


# In[66]:


# Check recodes
recodes = ['var234', 'var234r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[67]:


# Recode var235 into range (for graphing)
df_merged['var235r'] = np.where(df_merged['var235']==10741, 1, 
                              (np.where(df_merged['var235']==10742,2,
                                       (np.where(df_merged['var235']==10743,3,4)))))


# In[68]:


# Check recodes
recodes = ['var235', 'var235r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[70]:


# Recode var236 into range (for graphing)
df_merged['var236r'] = np.where(df_merged['var236']==10745, 1, 
                              (np.where(df_merged['var236']==10746,2,
                                       (np.where(df_merged['var236']==10747,3,
                                                (np.where(df_merged['var236']==10748,4,
                                                         (np.where(df_merged['var236']==10749,5,6)))))))))


# In[71]:


# Check recodes
recodes = ['var236', 'var236r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[72]:


# Recode var246 into range (for graphing)
df_merged['var246r'] = np.where(df_merged['var246']==10787, 1, 
                              (np.where(df_merged['var246']==10788,2,3)))


# In[73]:


# Check recodes
recodes = ['var246', 'var246r']
checkslice = df_merged[recodes]
checkslice_counts = checkslice.apply(lambda x: x.value_counts()).T.stack()
print(checkslice_counts)


# In[80]:


# Save records with recoded variables as a new csv file
df_merged.to_csv("OpioidsMerged_Recoded.csv", index=False)


# In[74]:


# Select features of interest and create dataframe for cluster analysis
features = ['var9r', 'var11r', 'var12r', 'var13r', 'educrec', 'incomerec', 'var16r', 'politrec', 'var217r',
            'var230r', 'var231r', 'var232r', 'var233r', 'var234r', 'var235r', 'var236r', 'var237rec', 'var238rec', 
            'var239rec', 'var240rec', 'var241rec', 'var242rec', 'var243rec', 'var244rec', 'var245rec', 'var246r', 
            'var247rec','var248rec', 'var249rec', 'var250rec', 'var251rec', 'var252rec', 'var253rec', 'var254rec',
            'var255rec', 'CT']
cslice = df_merged[features]
cslice.head()


# In[75]:


# Check variable frequencies
cslice_counts = cslice.apply(lambda x: x.value_counts()).T.stack()
cslice_counts.head()


# In[76]:


print(cslice_counts['CT'])


# In[77]:


cslice_counts.tail()


# In[83]:


cluster_range = range( 1, 11 )


# In[84]:


for n_clusters in cluster_range:
    km = KModes(n_clusters, init='Huang', n_init=10, verbose=1)
    km.fit(cslice)


# In[86]:


# Plot costs by number of clusters
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [17513.0, 15391.0, 13947.0, 13507.0, 13236.0, 12803.0, 12625.0, 12467.0, 12292.0, 12101])
plt.xlabel('Clusters')
plt.ylabel('Costs')
plt.axis([0, 11, 12000.0, 18000.0])
plt.show()


# ## Evaluate 3 clusters

# In[87]:


# Calculate 3 clusters
km3 = KModes(n_clusters=3, init='Huang', n_init=10, verbose=1)
clusters = km3.fit_predict(cslice)
print(km3.cluster_centroids_)


# In[88]:


centroids = km3.cluster_centroids_
centroids


# In[93]:


# Create a DataFrame with a column for cluster number
def pd_centroids(featuresUsed, centroids):
        colNames = list(featuresUsed)
        colNames.append('prediction')
            
        # Zip with a column called 'prediction' (index)
        Z = [np.append(A, index) for index, A in enumerate(centroids)]
        
        # Convert to pandas data frame for plotting
        P = pd. DataFrame(Z, columns=colNames)
        P['prediction'] = P['prediction'].astype(int)
        return P


# In[121]:


# Create parallel plots
def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize=(15,8)).gca().axes.set_ylim([0,+8])
    plt.xticks(rotation=70)
    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')


# In[98]:


P=pd_centroids(features, centroids)
P


# In[103]:


# Save centroids as a new csv file
P.to_csv("OpioidsCentroids.csv", index=False)


# In[116]:


# Drop columns with no variance
P2 = P.loc[:, P.var() > 0.0]
P2


# In[122]:


# Plot clusters
parallel_plot(P2)

