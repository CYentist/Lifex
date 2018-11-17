---
title: First Python Program (C1W2)
date: 2018-11-6 00:23:23
categories:
 - Homework -Data
photos:
tags: DataAnalysis

---

```python
import pandas
import numpy

gapminder = pandas.read_csv('gapminder.csv', low_memory=False)

print(len(gapminder)) #number of obs
print(len(gapminder.columns)) #number of variables
```

    213
    16



```python

#Convert to numeric variable

gapminder['urbanrate']=gapminder['urbanrate'].convert_objects(convert_numeric=True)
gapminder['incomeperperson']=gapminder['incomeperperson'].convert_objects(convert_numeric=True)
gapminder['employrate']=gapminder['employrate'].convert_objects(convert_numeric=True)
gapminder['lifeexpectancy']=gapminder['lifeexpectancy'].convert_objects(convert_numeric=True)


#Too many values, group the numeric values
gp=gapminder.copy()

gp['urbanrate_freq'] = pd.qcut(gp['urbanrate'], 3, labels=["low", "medium", "high"])
gp['incomeperperson_freq'] = pd.qcut(gp['incomeperperson'], 3, labels=["low", "medium", "high"])
gp['employrate_freq'] = pd.qcut(gp['employrate'], 3, labels=["low", "medium", "high"])
gp['lifeexpectancy_freq'] = pd.qcut(gp['lifeexpectancy'], 3, labels=["low", "medium", "high"])

```



```python

#Check frequency distribusion
print ('counts for urbanrate, 2008 urban population')
c1 = gp['urbanrate_freq'].value_counts(sort=False)
print(c1)

print ('percentages for urbanrate, 2008 urban population')
p1 = gp['urbanrate_freq'].value_counts(sort=False, normalize=True)
print (p1)

print ('counts for incomeperperson, 2010 Gross Domestic Product per capita in constant 2000 US$')
c2 = gp['incomeperperson_freq'].value_counts(sort=False)
print(c2)

print ('percentages for incomeperperson, 2010 Gross Domestic Product per capita in constant 2000 US$')
p2 = gp['incomeperperson_freq'].value_counts(sort=False, normalize=True)
print (p2)

print ('counts for employrate, 2007 total employees age 15+ (% of population)')
c3 = gp['employrate_freq'].value_counts(sort=False)
print(c3)

print ('percentages for employrate, 2007 total employees age 15+ (% of population)')
p3 = gp['employrate_freq'].value_counts(sort=False, normalize=True)
print (p3)
```

    counts for urbanrate, 2008 urban population
    low       68
    medium    67
    high      68
    Name: urbanrate_freq, dtype: int64
    percentages for urbanrate, 2008 urban population
    low       0.334975
    medium    0.330049
    high      0.334975
    Name: urbanrate_freq, dtype: float64
    counts for incomeperperson, 2010 Gross Domestic Product per capita in constant 2000 US$
    low       64
    medium    63
    high      63
    Name: incomeperperson_freq, dtype: int64
    percentages for incomeperperson, 2010 Gross Domestic Product per capita in constant 2000 US$
    low       0.336842
    medium    0.331579
    high      0.331579
    Name: incomeperperson_freq, dtype: float64
    counts for employrate, 2007 total employees age 15+ (% of population)
    low       60
    medium    59
    high      59
    Name: employrate_freq, dtype: int64
    percentages for employrate, 2007 total employees age 15+ (% of population)
    low       0.337079
    medium    0.331461
    high      0.331461
    Name: employrate_freq, dtype: float64



```python

#  using the 'bygroup' function
ct1= gp.groupby('urbanrate_freq').mean()
print (ct1)

ct2= gp.groupby('incomeperperson_freq').mean()
print (ct2)

ct3= gp.groupby('employrate_freq').mean()
print (ct3)

ct4= gp.groupby('lifeexpectancy_freq').mean()
print (ct4)
```

                    incomeperperson  lifeexpectancy  employrate  urbanrate
    urbanrate_freq                                                        
    low                 2992.557932       61.946859   64.687931  29.284412
    medium              4988.756425       70.674262   55.053448  57.622388
    high               18384.793740       76.720441   57.178947  83.413824
                          incomeperperson  lifeexpectancy  employrate  urbanrate
    incomeperperson_freq                                                        
    low                        568.490443       60.721397   64.003279  36.652813
    medium                    3111.048549       71.514475   54.475472  57.481935
    high                     22673.081072       78.296192   57.703846  75.197143
                     incomeperperson  lifeexpectancy  employrate  urbanrate
    employrate_freq                                                        
    low                  6856.503278       72.095655   47.328333  63.393818
    medium               9369.636740       71.884593   58.606780  59.303390
    high                 6464.009959       64.708610   70.164407  47.670508
                         incomeperperson  lifeexpectancy  employrate  urbanrate
    lifeexpectancy_freq                                                        
    low                       934.992977       58.014609   64.106452   37.98750
    medium                   3363.599929       72.486429   54.509091   57.58381
    high                    18423.475758       78.802234   56.972881   73.05541



I checked the frequency distribution of urban rate, income per person, employ rate and life expectancy and there association.

1.  In the frequency table of urban rate, lower urban rate tend to have lower income per person, employ rate and employ rate
2.  In the frequency table of income per person, lower incomer per person tend to have lower life expectancy, employ rate and urban rate
3.  In the frequency table of employ rate, lower employ rate tend to have higher urban rate and higher life expectancy, however, both low and high employ rate have lower income per person than medium employ rate
4.  In the frequency table of life expectancy, lower life expectancy tend to have lower income per person, lower urban rate but higher employ rate.

In this research, it seems like urban rate, life expectancy and income per person may have a positive correlation, but employ rate of a country may have any association with any one of the other three variable.