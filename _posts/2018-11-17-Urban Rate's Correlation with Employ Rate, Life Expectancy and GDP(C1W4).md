---
title: Urban Rate's Correlation with Employ Rate, Life Expectancy and GDP(C1W4)
date: 2018-11-17 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis

---

```python

import pandas as pd
import numpy
import seaborn
import matplotlib.pyplot as plt

data = pd.read_csv('gapminder.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)


data['urbanrate'] = data['urbanrate'].convert_objects(convert_numeric=True)
data['incomeperperson'] = data['incomeperperson'].convert_objects(convert_numeric=True)
data['employrate'] = data['employrate'].convert_objects(convert_numeric=True)
data['lifeexpectancy'] = data['lifeexpectancy'].convert_objects(convert_numeric=True)

```






```python
# Create Secondary variables

gp=data.copy()
#new urbanrate_freq variable, categorical 1 through 6
def urbanrate_freq (row):
   if row['urbanrate'] > 75 :
      return 'high'
   elif row['urbanrate'] > 50 :
      return 'medium'
   elif row['urbanrate'] > 25 :
      return 'low'
   else  :
      return 'very low'

gp['urbanrate_freq'] = gp.apply (lambda row: urbanrate_freq (row),axis=1)

seaborn.countplot(x="urbanrate_freq", data=gp)
plt.xlabel('Urbanization Rate')
plt.title('Grouped Urbanization Rate Distribution')

```




    Text(0.5, 1.0, 'Grouped Urbanization Rate Distribution')




![png](http://img.luhaoip.com/images/2018-11-17-080248.jpg)



```python
#variation
gp['urbanrate'].var()
```




    568.580812954202




```python

#Histogram of Urabanrate 
gp.hist(column='urbanrate', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
plt.title(u'Urbanrate')
plt.show()
```


![png](http://img.luhaoip.com/images/2018-11-17-080245.jpg)


Urban rate seems like have a normal distribution with large variation


```python

def employrate_freq (row):
   if row['employrate'] > 75 :
      return 'high'
   elif row['employrate'] > 50 :
      return 'medium'
   elif row['employrate'] > 25 :
      return 'low'
   else  :
      return 'very low'

gp['employrate_freq'] = gp.apply (lambda row: employrate_freq (row),axis=1)
# first change the variable format to categorical if you haven’t already done so
gp['employrate_freq'] = gp['employrate_freq'].astype('category')

seaborn.countplot(x="employrate_freq", data=gp)

plt.xlabel('Employe Rate')
plt.title('Employe Rate Distribution')
```




    Text(0.5, 1.0, 'Employe Rate Distribution')




![png](http://img.luhaoip.com/images/2018-11-17-080244.jpg)



```python

#Histogram of Employe Rate' 
gp.hist(column='employrate', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
plt.title(u'Employrate')
plt.show()

```


![png](http://img.luhaoip.com/images/2018-11-17-080240.jpg)



```python
#variation
gp['employrate'].var()
```




    110.65892206783462



Employ rate have a normal distribution


```python
#Histogram of Income per person' 
gp.hist(column='incomeperperson', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
plt.title(u'Income per person')
plt.show()
```


![png](http://img.luhaoip.com/images/2018-11-17-080249.jpg)


Income per person have a exponential distribution


```python

#Histogram of life expectancy' 
gp.hist(column='lifeexpectancy', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
plt.title(u'Life Expectancy')
plt.show()
```


![png](http://img.luhaoip.com/images/2018-11-17-080246.jpg)


Life expectancy's distribution looks like a exponential distribution but with a lower variation


```python

#Scatter Plot

scat1 = seaborn.regplot(x="urbanrate", y="employrate",  data=gp)
plt.xlabel('Urban Rate')
plt.ylabel('Employ Rate')
plt.title('Scatterplot for the Association Between Urban Rate and Employ Rate')

```

    /Users/CY/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    Text(0.5, 1.0, 'Scatterplot for the Association Between Urban Rate and Employ Rate')




![png](http://img.luhaoip.com/images/2018-11-17-080243.jpg)


Urban rate and employ rate a negative corelated


```python

scat2 = seaborn.regplot(x="urbanrate", y="lifeexpectancy",  data=gp)
plt.xlabel('Urban Rate')
plt.ylabel('Life Expectancy')
plt.title('Scatterplot for the Association Between Urban Rate and Life Expectancy')

```




    Text(0.5, 1.0, 'Scatterplot for the Association Between Urban Rate and Life Expectancy')




![png](http://img.luhaoip.com/images/2018-11-17-080247.jpg)


Urban rate and life expectancy are positive correlated


```python

scat3 = seaborn.regplot(x="urbanrate", y="incomeperperson",  data=gp)
plt.xlabel('Urban Rate')
plt.ylabel('incomeperperson')
plt.title('Scatterplot for the Association Between Urban Rate and GDP')

```




    Text(0.5, 1.0, 'Scatterplot for the Association Between Urban Rate and GDP')




![png](http://img.luhaoip.com/images/2018-11-17-80247.jpg)


Urban rate and income per person are positive correlated


```python
# Quartile Split
print ('Income per person - 4 categories - quartiles')
gp['INCOMEGRP4']=pd.qcut(gp['incomeperperson'], 4, labels=["1=25th%tile","2=50%tile","3=75%tile","4=100%tile"])
c10 = gp['INCOMEGRP4'].value_counts(sort=False, dropna=True)
print(c10)


c11= gp.groupby('INCOMEGRP4').size()
print (c11)

```

    Income per person - 4 categories - quartiles
    1=25th%tile    48
    2=50%tile      47
    3=75%tile      47
    4=100%tile     48
    Name: INCOMEGRP4, dtype: int64
    INCOMEGRP4
    1=25th%tile    48
    2=50%tile      47
    3=75%tile      47
    4=100%tile     48
    dtype: int64



