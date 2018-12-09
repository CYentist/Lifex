---
title:  Testing a basic linear regression(C3W2)
date: 2018-12-09 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis


---



```python

import numpy as numpyp
import pandas as pd
import statsmodels.api
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt


# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pd.read_csv('gapminder.csv')

# convert variables to numeric format using convert_objects function
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')

```

Center explanatory variable


```python
data['urbanrate_center'] = data['urbanrate'] - data['urbanrate'].mean()

m1 = data['urbanrate_center'].mean()
print(m1)
```

    1.8446109445594718e-14



```python
scat1 = seaborn.regplot(x="urbanrate_center", y="incomeperperson", scatter=True, data=data)
plt.xlabel('Urbanization Rate')
plt.ylabel('Income Per Person')
plt.title ('Scatterplot for the Association Between Urban Rate and Income Per Person')
print(scat1)

print ("OLS regression model for the association between urban rate and Income Per Person")
reg1 = smf.ols('incomeperperson ~ urbanrate_center', data=data).fit()
print (reg1.summary())
```

    AxesSubplot(0.125,0.125;0.775x0.755)
    OLS regression model for the association between urban rate and Income Per Person
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        incomeperperson   R-squared:                       0.240
    Model:                            OLS   Adj. R-squared:                  0.236
    Method:                 Least Squares   F-statistic:                     59.11
    Date:                Sun, 09 Dec 2018   Prob (F-statistic):           8.20e-13
    Time:                        16:09:03   Log-Likelihood:                -2050.0
    No. Observations:                 189   AIC:                             4104.
    Df Residuals:                     187   BIC:                             4110.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept         8906.7324    908.718      9.801      0.000    7114.076    1.07e+04
    urbanrate_center   295.2035     38.395      7.689      0.000     219.461     370.946
    ==============================================================================
    Omnibus:                      185.190   Durbin-Watson:                   2.193
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3954.152
    Skew:                           3.758   Prob(JB):                         0.00
    Kurtosis:                      24.109   Cond. No.                         23.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



![png](http://img.luhaoip.com/images/2018-12-09-082253.jpg)


The correlation coefficient is 295.2035 and the intercepte is 8906.73. P<0.001. The R square is 0.240, which mean 24% of the response variable - income per person could be explained by the explanartory variable - the urban rate.
