---
title:  K-means Clustering(C4W4)
date: 2018-12-28 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis

---



```python


#from pandas import Series, DataFrame
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans



data = pd.read_csv('gapminder.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)

```


```python

# convert variables to numeric format using convert_objects function
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')
data['femaleemployrate'] = pd.to_numeric(data['femaleemployrate'], errors='coerce')
data['internetuserate'] = pd.to_numeric(data['internetuserate'], errors='coerce')
data['alcconsumption'] = pd.to_numeric(data['alcconsumption'], errors='coerce')
data['armedforcesrate'] = pd.to_numeric(data['armedforcesrate'], errors='coerce')
data['breastcancerper100th'] = pd.to_numeric(data['breastcancerper100th'], errors='coerce')
data['co2emissions'] = pd.to_numeric(data['co2emissions'], errors='coerce')
data['polityscore'] = pd.to_numeric(data['polityscore'], errors='coerce')
data['relectricperperson'] = pd.to_numeric(data['relectricperperson'], errors='coerce')
data['suicideper100th'] = pd.to_numeric(data['suicideper100th'], errors='coerce')

#drop missing values
data_clean = data.dropna()

#define cluster variables
clustor = data_clean[['incomeperperson','alcconsumption','armedforcesrate','breastcancerper100th','co2emissions','polityscore','relectricperperson','suicideper100th','urbanrate', 'internetuserate','employrate', 'femaleemployrate','lifeexpectancy']]


clustor_p=clustor.copy()

#normalized variables
from sklearn import preprocessing
clustor_p['incomeperperson']=preprocessing.scale(clustor_p['incomeperperson'].astype('float64'))
clustor_p['employrate']=preprocessing.scale(clustor_p['employrate'].astype('float64'))
clustor_p['lifeexpectancy']=preprocessing.scale(clustor_p['lifeexpectancy'].astype('float64'))
clustor_p['urbanrate']=preprocessing.scale(clustor_p['urbanrate'].astype('float64'))
clustor_p['femaleemployrate']=preprocessing.scale(clustor_p['femaleemployrate'].astype('float64'))
clustor_p['internetuserate']=preprocessing.scale(clustor_p['internetuserate'].astype('float64'))
clustor_p['alcconsumption']=preprocessing.scale(clustor_p['alcconsumption'].astype('float64'))
clustor_p['armedforcesrate']=preprocessing.scale(clustor_p['armedforcesrate'].astype('float64'))
clustor_p['breastcancerper100th']=preprocessing.scale(clustor_p['breastcancerper100th'].astype('float64'))
clustor_p['polityscore']=preprocessing.scale(clustor_p['polityscore'].astype('float64'))
clustor_p['relectricperperson']=preprocessing.scale(clustor_p['relectricperperson'].astype('float64'))
clustor_p['suicideper100th']=preprocessing.scale(clustor_p['suicideper100th'].astype('float64'))

```


```python

# split data into train and test sets
clus_train, clus_test = train_test_split(clustor_p, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
#    store  cluster number of each analysis
    clusassign=model.predict(clus_train)
#    caluculate average distance
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

```


```python

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

```




    Text(0.5, 1.0, 'Selecting k with the Elbow Method')




![png](http://img.luhaoip.com/images/2018-12-28-081156.jpg)


A K-means cluster analysis was conducted to the gapminder dataset. Test 1-9 different cluster method and plot the number of clusters and average distance above. Seems like 2 and 4 clusters is making more sense than other solutions.

Due to the data set is not a very large, 2 clusters is a better classification method. 



```python
model2=KMeans(n_clusters=2)
model2.fit(clus_train)
clusassign2=model2.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
#return the first two cononical variable
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)

#plot first cononical variable in x, second in Y
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model2.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()
```


![png](http://img.luhaoip.com/images/2018-12-28-081201.jpg)



```python

"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model2.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

```




    0    78
    1     6
    Name: cluster, dtype: int64



6 counted as cluster 1, and 78 counted as cluster 0



```python

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
```

    Clustering variable means by cluster
                 index  incomeperperson  alcconsumption  armedforcesrate  \
    cluster                                                                
    0       109.128205        -0.125447       -0.113195         0.013388   
    1        92.166667         1.117909        0.304675        -0.562392   
    
             breastcancerper100th       co2emissions  polityscore  \
    cluster                                                         
    0                   -0.121701  2402952444.444445    -0.114736   
    1                    0.619192 54160864388.888832     0.399074   
    
             relectricperperson  suicideper100th  urbanrate  internetuserate  \
    cluster                                                                    
    0                 -0.140929        -0.018180  -0.037066        -0.131948   
    1                  0.228978         0.649822   0.121132         0.826089   
    
             employrate  femaleemployrate  lifeexpectancy  
    cluster                                                
    0          0.000277         -0.038404       -0.152284  
    1          0.055545          0.204108        0.695874  



```python

# validate clusters in training data by examining cluster differences in income per person using ANOVA
# first have to merge incomer per person with clustering variables and cluster assignment data 
inc_data=data_clean['incomeperperson']
# split income data into train and test sets
inc_train, inc_test = train_test_split(inc_data, test_size=.3, random_state=123)
inc_train1=pd.DataFrame(inc_train)
inc_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(inc_train1, merged_train, on='index')
sub1 = merged_train_all[['incomeperperson_x', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

incmod = smf.ols(formula='incomeperperson_x ~ C(cluster)', data=sub1).fit()
print (incmod.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      incomeperperson_x   R-squared:                       0.102
    Model:                            OLS   Adj. R-squared:                  0.091
    Method:                 Least Squares   F-statistic:                     9.294
    Date:                Fri, 28 Dec 2018   Prob (F-statistic):            0.00309
    Time:                        16:01:48   Log-Likelihood:                -892.55
    No. Observations:                  84   AIC:                             1789.
    Df Residuals:                      82   BIC:                             1794.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept        6765.8152   1141.772      5.926      0.000    4494.467    9037.163
    C(cluster)[T.1]  1.302e+04   4272.120      3.049      0.003    4525.543    2.15e+04
    ==============================================================================
    Omnibus:                       32.819   Durbin-Watson:                   2.114
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               55.117
    Skew:                           1.626   Prob(JB):                     1.08e-12
    Kurtosis:                       5.274   Cond. No.                         3.90
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The p-value is 0.003, which is enough to reject the H0. The income per person is significant different between the two groups.


```python

#Incomer per person of each group is significant different

print ('means for income per person by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

```

    means for income per person by cluster
             incomeperperson_x
    cluster                   
    0              6765.815200
    1             19789.965549



```python
print ('standard deviations for income per person by cluster')
m2= sub1.groupby('cluster').std()
print (m2)
```

    standard deviations for income per person by cluster
             incomeperperson_x
    cluster                   
    0              9660.818238
    1             15176.194396


**Summary**

Afer the K-meas analysis, all the country could be seperated into two different group, and each group are having different incomer per person comparing with the other group. The higher income group also have high life expectancy, urban rate and internet use rate.

