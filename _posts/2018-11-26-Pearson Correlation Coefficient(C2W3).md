---
title: Pearson Correlation Coefficient(C2W3)
date: 2018-11-26 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis


---



```python
import pandas as pd
import numpy
import seaborn
import scipy
import matplotlib.pyplot as plt

data = pd.read_csv('gapminder.csv', low_memory=False)

#Setting varibles to numeric
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')

data_clean=data.dropna()

```

```python
scat1 = seaborn.regplot(x="urbanrate", y="incomeperperson", fit_reg=True, data=data)
plt.xlabel('Urban Rate')
plt.ylabel('Income per person')
plt.title('Scatterplot for the Association Between Urban Rate and Income per person')


print ('association between urbanrate and Income per person')
#Pearson correlation coefficien, P-value
print (scipy.stats.pearsonr(data_clean['urbanrate'], data_clean['incomeperperson']))

```

```
association between urbanrate and Income per person
(0.6185795259780851, 6.6444000821161035e-19)
```



![png](http://img.luhaoip.com/images/2018-11-27-225714.jpg)

Pearson correlation coefficien is 0.619, and p-value is very small, so there is a positive relationship between urban rate and income per-person 



```python
scat2 = seaborn.regplot(x="urbanrate", y="lifeexpectancy", fit_reg=True, data=data)
plt.xlabel('Urban Rate')
plt.ylabel('lifeexpectancy')
plt.title('Scatterplot for the Association Between Urban Rate and lifeexpectancy')


print ('association between urbanrate and lifeexpectancy')
print (scipy.stats.pearsonr(data_clean['urbanrate'], data_clean['lifeexpectancy']))

```

```
association between urbanrate and lifeexpectancy
(0.6666916935934922, 1.0770484556505072e-22)
```



![png](http://img.luhaoip.com/images/2018-11-27-225715.jpg)

Pearson correlation coefficien is 0.667, and p-value is very small, so there is a positive relationship between urban rate and life expectancy



```python
scat3 = seaborn.regplot(x="urbanrate", y="employrate", fit_reg=True, data=data)
plt.xlabel('Urban Rate')
plt.ylabel('employrate')
plt.title('Scatterplot for the Association Between Urban Rate and employrate')


print ('association between urbanrate and employrate')
print (scipy.stats.pearsonr(data_clean['urbanrate'], data_clean['employrate']))

```

```
association between urbanrate and employrate
(-0.3298643940830164, 1.4248655106937845e-05)
```



![png](http://img.luhaoip.com/images/2018-11-27-225713.jpg)

Pearson correlation coefficien is -0.330, and p-value is very small, so there is a modest negative relationship between urban rate and employ rate



