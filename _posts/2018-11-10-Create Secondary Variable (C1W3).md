---
title: Create Secondary Variable
date: 2018-11-10 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis

---



```python
import pandas as pd
import numpy

data = pd.read_csv('gapminder.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)

data['urbanrate'] = data['urbanrate'].convert_objects(convert_numeric=True)
data['incomeperperson'] = data['incomeperperson'].convert_objects(convert_numeric=True)
data['employrate'] = data['employrate'].convert_objects(convert_numeric=True)
data['lifeexpectancy'] = data['lifeexpectancy'].convert_objects(convert_numeric=True)


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


gp['incomeperperson_freq'] = pd.qcut(gp['incomeperperson'], 3, labels=["low", "medium", "high"])
gp['lifeexpectancy_freq'] = pd.qcut(gp['lifeexpectancy'], 3, labels=["low", "medium", "high"])



#frequency distributions for primary and secondary ethinciity variables
print ('counts for urbanrate_freq')
c1 = gp['urbanrate_freq'].value_counts(sort=False)
print(c1)

print ('counts for employrate_freq')
c2 = gp['employrate_freq'].value_counts(sort=False)
print(c2)

print ('counts for incomeperperson_freq')
c3 = gp['incomeperperson_freq'].value_counts(sort=False)
print(c3)

print ('counts for lifeexpectancy_freq')
c4 = gp['lifeexpectancy_freq'].value_counts(sort=False)
print(c4)
```

    counts for urbanrate_freq
    very low    32
    high        48
    medium      74
    low         59
    Name: urbanrate_freq, dtype: int64
    counts for employrate_freq
    high         14
    low          37
    medium      127
    very low     35
    Name: employrate_freq, dtype: int64
    counts for incomeperperson_freq
    low       64
    medium    63
    high      63
    Name: incomeperperson_freq, dtype: int64
    counts for lifeexpectancy_freq
    low       64
    medium    63
    high      64
    Name: lifeexpectancy_freq, dtype: int64

1. About missing values, since the variables I choose don't have any missing value, and most of them are numeric values, I don't need to coding any missing value
2. I created four secondary variables from the numeric variables. Divided them into 3 or 4 levels then checked the frequency distribution of each secondary variable.
3. Foundings:
   1. Most countries have medium urban rate. This is lower than I expected
   2. Most countries' employ rate are between 50% to 75%, which is medium.