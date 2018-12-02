---
title: Chi Square and Post hoc analysis(C2W2)
date: 2018-11-23 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis
---





```python
import pandas as pd
import numpy
import scipy.stats
import seaborn
import matplotlib.pyplot as plt

data = pd.read_csv('gapminder.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)



# new code setting variables you will be working with to numeric
data['urbanrate'] = pandas.to_numeric(data['urbanrate'], errors='coerce')
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors='coerce')
data['employrate'] = pandas.to_numeric(data['employrate'], errors='coerce')
data['lifeexpectancy'] = pandas.to_numeric(data['lifeexpectancy'], errors='coerce')

```

```python
# Create Secondary variables

gp=data.copy()
#new urbanrate_freq variable, categorical 1 through 6
def urbanrate_freq (row):
   if row['urbanrate'] > 66 :
      return 'high'
   elif row['urbanrate'] > 33 :
      return 'medium'
   else  :
      return 'low'

gp['urbanrate_freq'] = gp.apply (lambda row: urbanrate_freq (row),axis=1)

# set variable types 
gp["urbanrate_freq"] = gp["urbanrate_freq"].astype('category')

seaborn.countplot(x="urbanrate_freq", data=gp)
plt.xlabel('Urbanization Rate')
plt.title('Grouped Urbanization Rate Distribution')
```



```
Text(0.5, 1.0, 'Grouped Urbanization Rate Distribution')
```



![png](http://img.luhaoip.com/images/2018-11-27-144642.jpg)



```python
def employrate_freq (row):
   if row['employrate'] > 50 :
      return 'high'
   else  :
      return 'low'

gp['employrate_freq'] = gp.apply (lambda row: employrate_freq (row),axis=1)
# first change the variable format to categorical if you haven’t already done so
gp['employrate_freq'] = gp['employrate_freq'].astype('category')

seaborn.countplot(x="employrate_freq", data=gp)

plt.xlabel('Employe Rate')
plt.title('Employe Rate Distribution')
```



```
Text(0.5, 1.0, 'Employe Rate Distribution')
```



![png](http://img.luhaoip.com/images/2018-11-27-144640.jpg)



```python
# contingency table of observed counts
#First is explanatory variable, second is response variable
ct1=pandas.crosstab(gp['urbanrate_freq'] ,gp['employrate_freq']  )
print (ct1)

seaborn.catplot(x="urbanrate_freq", y="employrate", data=gp, kind="bar", ci=None)
plt.xlabel('Urban Rate')
plt.ylabel('Employ Rate')

# column percentages
# axis=0 means row, axis =1 means column
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)
```

```
employrate_freq  high  low
urbanrate_freq            
high               52   27
low                32   20
medium             57   25
employrate_freq     high      low
urbanrate_freq                   
high            0.368794 0.375000
low             0.226950 0.277778
medium          0.404255 0.347222
chi-square value, p value, expected counts
(0.9120225646853408, 0.6338066862339891, 2, array([[52.29577465, 26.70422535],
       [34.42253521, 17.57746479],
       [54.28169014, 27.71830986]]))
```



![png](http://img.luhaoip.com/images/2018-11-27-144641.jpg)



**P > 0.05 means groups are equal with one another**

