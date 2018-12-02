---
title: Finding the Moderator(C2W4)
date: 2018-11-29 00:23:23
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
data['internetuserate'] = pd.to_numeric(data['internetuserate'], errors='coerce')


data_clean=data.dropna()

print (scipy.stats.pearsonr(data_clean['internetuserate'], data_clean['employrate']))

scat = seaborn.regplot(x="internetuserate", y="employrate", data=data_clean)
plt.xlabel('Internet Use Rate')
plt.ylabel('Employe Rate')
plt.title('Scatterplot for the Association Between Internet Use Rate and Employe Rate')
print (scat)
```

    (-0.20084402120259573, 0.009916806362522712)



![png](http://img.luhaoip.com/images/2018-12-02-035921.jpg)


Internet use rate and employ rate are negative related.


Check if Urban Rate is a moderator


```python

data_clean['urbanrate_freq'] = pd.qcut(data_clean['urbanrate'], 3, labels=["low", "medium", "high"])

sub1=data_clean[(data_clean['urbanrate_freq']=='high')]
sub2=data_clean[(data_clean['urbanrate_freq']== 'medium')]
sub3=data_clean[(data_clean['urbanrate_freq']== 'low')]

print ('association between urbanrate and incomeperperson for High employrate countries')
print (scipy.stats.pearsonr(sub1['internetuserate'], sub1['employrate']))
print ('       ')

scat1 = seaborn.regplot(x="internetuserate", y="employrate", data=sub1)
plt.xlabel('Internet Use Rate')
plt.ylabel('Employe Rate')
plt.title('Scatterplot for the Association Between Internet Use Rate and Employe Rate for High Urban Rate Countries')
print (scat1)
```





    association between urbanrate and incomeperperson for High employrate countries
    (0.4077657606064094, 0.0020007024183570136)



![png](http://img.luhaoip.com/images/2018-12-02-035919.jpg)


Internet use rate and employ rate are positive related in High urban rate countries


```python

print ('association between urbanrate and incomeperperson for Medium employrate countries')
print (scipy.stats.pearsonr(sub2['internetuserate'], sub2['employrate']))
print ('       ')

scat2 = seaborn.regplot(x="internetuserate", y="employrate", data=sub2)
plt.xlabel('Internet Use Rate')
plt.ylabel('Employe Rate')
plt.title('Scatterplot for the Association Between Internet Use Rate and Employe Rate for Medium Urban Rate Countries')
print (scat2)
```

    association between urbanrate and incomeperperson for Medium employrate countries
    (-0.27136019469548, 0.04716247368667033)
           
    AxesSubplot(0.125,0.125;0.775x0.755)



![png](http://img.luhaoip.com/images/2018-12-02-035922.jpg)


Internet use rate and employ rate are negative related in Medium urban rate countries


```python

print ('association between urbanrate and incomeperperson for Low employrate countries')
print (scipy.stats.pearsonr(sub3['internetuserate'], sub3['employrate']))
print ('       ')

scat3 = seaborn.regplot(x="internetuserate", y="employrate", data=sub3)
plt.xlabel('Internet Use Rate')
plt.ylabel('Employe Rate')
plt.title('Scatterplot for the Association Between Internet Use Rate and Employe Rate for Low Urban Rate Countries')
print (scat3)
```

    association between urbanrate and incomeperperson for Low employrate countries
    (-0.2609425804677143, 0.05432506564141465)
           
    AxesSubplot(0.125,0.125;0.775x0.755)



![png](http://img.luhaoip.com/images/2018-12-02-035920.jpg)


P>0.05, Internet use rate and employ rate are not related in Low urban rate countries

**Conclusion**:

Internet use rate and employe rate are having a weak negative correlation in medium urban rate countries, and a high positive corrlation in high urban rate countries. **So urban rate is the moderator of internet use rate and employe rate relationship.**

