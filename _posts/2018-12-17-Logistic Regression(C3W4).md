---
title:  Logistic Regression
date: 2018-12-17 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis



---



```python

import numpy
import pandas as pd
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 


# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pd.read_csv('gapminder.csv')


# convert variables to numeric format using convert_objects function
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['femaleemployrate'] = pd.to_numeric(data['femaleemployrate'], errors='coerce')
data['internetuserate'] = pd.to_numeric(data['internetuserate'], errors='coerce')

sub1 = data[['urbanrate', 'internetuserate','employrate', 'femaleemployrate','lifeexpectancy','incomeperperson']].dropna()

sub1['urbanrate_center'] = (sub1['urbanrate'] - sub1['urbanrate'].mean())
sub1['employrate_center'] = (sub1['employrate'] - sub1['employrate'].mean())
sub1['lifeexpectancy_center'] = (sub1['lifeexpectancy'] - sub1['lifeexpectancy'].mean())
sub1['femaleemployrate_center'] = (sub1['femaleemployrate'] - sub1['femaleemployrate'].mean())
sub1['internetuserate_center'] = (sub1['internetuserate'] - sub1['internetuserate'].mean())

```

High-income economies are those in which 2016 GNI per capita was $12,235 or more.(World Bank https://en.wikipedia.org/wiki/World_Bank_high-income_economy)


```python

#dependent variable tranformation
def HIINCOME (x):
   if x['incomeperperson']>= 12235:
      return 1
   else: 
      return 0
  
sub1['HIINCOME'] = sub1.apply (lambda x: HIINCOME (x), axis=1)


seaborn.countplot(x="HIINCOME", data=sub1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c23109208>




![png](http://img.luhaoip.com/images/2018-12-17-151829.jpg)



```python

#Logistic regression

lreg1 = smf.logit(formula = 'HIINCOME ~ urbanrate_center', data = sub1).fit()
print (lreg1.summary())

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))
```

    Optimization terminated successfully.
             Current function value: 0.327022
             Iterations 8
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               HIINCOME   No. Observations:                  164
    Model:                          Logit   Df Residuals:                      162
    Method:                           MLE   Df Model:                            1
    Date:                Mon, 17 Dec 2018   Pseudo R-squ.:                  0.3692
    Time:                        22:39:30   Log-Likelihood:                -53.632
    converged:                       True   LL-Null:                       -85.025
                                            LLR p-value:                 2.303e-15
    ====================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept           -2.4208      0.397     -6.096      0.000      -3.199      -1.643
    urbanrate_center     0.1005      0.018      5.509      0.000       0.065       0.136
    ====================================================================================
                      Lower CI  Upper CI   OR
    Intercept             0.04      0.19 0.09
    urbanrate_center      1.07      1.15 1.11



```python

# logistic regression with social phobia and depression
lreg2 = smf.logit(formula = 'HIINCOME ~ urbanrate_center + internetuserate_center', data = sub1).fit()
print (lreg2.summary())


# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))
```

    Optimization terminated successfully.
             Current function value: 0.171005
             Iterations 9
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               HIINCOME   No. Observations:                  164
    Model:                          Logit   Df Residuals:                      161
    Method:                           MLE   Df Model:                            2
    Date:                Mon, 17 Dec 2018   Pseudo R-squ.:                  0.6702
    Time:                        22:39:41   Log-Likelihood:                -28.045
    converged:                       True   LL-Null:                       -85.025
                                            LLR p-value:                 1.794e-25
    ==========================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 -4.4121      0.887     -4.972      0.000      -6.151      -2.673
    urbanrate_center           0.0780      0.026      2.956      0.003       0.026       0.130
    internetuserate_center     0.1109      0.023      4.800      0.000       0.066       0.156
    ==========================================================================================
    
    Possibly complete quasi-separation: A fraction 0.18 of observations can be
    perfectly predicted. This might indicate that there is complete
    quasi-separation. In this case some parameters will not be identified.
                            Lower CI  Upper CI   OR
    Intercept                   0.00      0.07 0.01
    urbanrate_center            1.03      1.14 1.08
    internetuserate_center      1.07      1.17 1.12


95% confident interval of urban rate after controling internet use rate is 1.03 - 1.14, 95% confident interval of internet use rate after controling urban rate is 1.17 - 1.12.


```python

# Add a potential confounder

# logistic regression with social phobia and depression
lreg3 = smf.logit(formula = 'HIINCOME ~ urbanrate_center + internetuserate_center+femaleemployrate_center', data = sub1).fit()
print (lreg3.summary())


# odd ratios with 95% confidence intervals
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))
```

    Optimization terminated successfully.
             Current function value: 0.163465
             Iterations 9
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               HIINCOME   No. Observations:                  164
    Model:                          Logit   Df Residuals:                      160
    Method:                           MLE   Df Model:                            3
    Date:                Mon, 17 Dec 2018   Pseudo R-squ.:                  0.6847
    Time:                        22:39:52   Log-Likelihood:                -26.808
    converged:                       True   LL-Null:                       -85.025
                                            LLR p-value:                 4.523e-25
    ===========================================================================================
                                  coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    Intercept                  -4.1986      0.883     -4.753      0.000      -5.930      -2.467
    urbanrate_center            0.0840      0.027      3.079      0.002       0.031       0.137
    internetuserate_center      0.1052      0.023      4.497      0.000       0.059       0.151
    femaleemployrate_center     0.0582      0.039      1.485      0.138      -0.019       0.135
    ===========================================================================================
    
    Possibly complete quasi-separation: A fraction 0.12 of observations can be
    perfectly predicted. This might indicate that there is complete
    quasi-separation. In this case some parameters will not be identified.
                             Lower CI  Upper CI   OR
    Intercept                    0.00      0.08 0.02
    urbanrate_center             1.03      1.15 1.09
    internetuserate_center       1.06      1.16 1.11
    femaleemployrate_center      0.98      1.14 1.06


Female employe rate is not significant relavent in this model, so it is may be a confounder.
