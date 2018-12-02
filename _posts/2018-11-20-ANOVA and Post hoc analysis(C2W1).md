---
title: ANOVA and Post hoc analysis(C2W1)
date: 2018-11-20 00:23:23
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



```
Text(0.5, 1.0, 'Grouped Urbanization Rate Distribution')
```



![png](http://img.luhaoip.com/images/2018-11-27-144102.jpg)



```python
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 


#ANOVA Test
#Test categorical variable urbanrate_freq and numeric variable variable incomeperperson
model1 = smf.ols(formula='incomeperperson ~ C(urbanrate_freq)', data=gp)
results1 = model1.fit()
print (results1.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:        incomeperperson   R-squared:                       0.299
Model:                            OLS   Adj. R-squared:                  0.288
Method:                 Least Squares   F-statistic:                     26.45
Date:                Sat, 24 Nov 2018   Prob (F-statistic):           2.70e-14
Time:                        14:19:51   Log-Likelihood:                -2052.8
No. Observations:                 190   AIC:                             4114.
Df Residuals:                     186   BIC:                             4127.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                      2.254e+04   1814.696     12.420      0.000     1.9e+04    2.61e+04
C(urbanrate_freq)[T.low]      -2.047e+04   2434.670     -8.409      0.000   -2.53e+04   -1.57e+04
C(urbanrate_freq)[T.medium]   -1.607e+04   2322.301     -6.922      0.000   -2.07e+04   -1.15e+04
C(urbanrate_freq)[T.very low] -1.756e+04   3143.146     -5.588      0.000   -2.38e+04   -1.14e+04
==============================================================================
Omnibus:                      180.154   Durbin-Watson:                   2.207
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3655.502
Skew:                           3.596   Prob(JB):                         0.00
Kurtosis:                      23.249   Cond. No.                         5.19
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

P<0.05 means we are confident enough to reject the hypothesis, which means *income per person are not equal in each urban rate group.*

```python
in1 = gp[['incomeperperson', 'lifeexpectancy','employrate','urbanrate_freq']].dropna()


#post hoc analysis - Tukey HSDT
mc1 = multi.MultiComparison(in1['incomeperperson'], in1['urbanrate_freq'])
res1 = mc1.tukeyhsd()
print(res1.summary())
```

```
    Multiple Comparison of Means - Tukey HSD,FWER=0.05    
==========================================================
group1  group2    meandiff     lower       upper    reject
----------------------------------------------------------
 high    low    -18497.7754 -23165.4478 -13830.1029  True 
 high   medium  -13407.9726  -17834.059  -8981.8862  True 
 high  very low -18892.9411 -25305.4375 -12480.4448  True 
 low    medium   5089.8028   1021.6127   9157.9929   True 
 low   very low  -395.1658   -6566.0684  5775.7369  False 
medium very low  -5484.9686 -11475.2202   505.283   False 
----------------------------------------------------------
```

Group "very low" doesn't have significant difference comparing with "low" and "medium"

```python
#Test categorical variable urbanrate_freq and numeric variable variable lifeexpectancy
model2 = smf.ols(formula='lifeexpectancy ~ C(urbanrate_freq)', data=gp)
results2 = model2.fit()
print (results2.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:         lifeexpectancy   R-squared:                       0.330
Model:                            OLS   Adj. R-squared:                  0.319
Method:                 Least Squares   F-statistic:                     30.71
Date:                Sat, 24 Nov 2018   Prob (F-statistic):           3.45e-16
Time:                        14:23:45   Log-Likelihood:                -666.41
No. Observations:                 191   AIC:                             1341.
Df Residuals:                     187   BIC:                             1354.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                        77.5444      1.251     61.989      0.000      75.077      80.012
C(urbanrate_freq)[T.low]        -14.0522      1.653     -8.503      0.000     -17.312     -10.792
C(urbanrate_freq)[T.medium]      -5.4110      1.571     -3.444      0.001      -8.510      -2.312
C(urbanrate_freq)[T.very low]   -13.7917      2.059     -6.699      0.000     -17.853      -9.730
==============================================================================
Omnibus:                        9.644   Durbin-Watson:                   1.799
Prob(Omnibus):                  0.008   Jarque-Bera (JB):               10.049
Skew:                          -0.562   Prob(JB):                      0.00658
Kurtosis:                       3.032   Cond. No.                         5.33
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

P<0.05 means we are confident enough to reject the hypothesis,which means life expectancy are not equal in each urban rate group.

```python
#post hoc analysis - Tukey HSDT
mc2 = multi.MultiComparison(in1['lifeexpectancy'], in1['urbanrate_freq'])
res2 = mc2.tukeyhsd()
print(res2.summary())
```

```
Multiple Comparison of Means - Tukey HSD,FWER=0.05
=================================================
group1  group2  meandiff  lower    upper   reject
-------------------------------------------------
 high    low    -15.4097 -19.7126 -11.1069  True 
 high   medium  -6.1169  -10.197  -2.0368   True 
 high  very low -18.3013 -24.2125  -12.39   True 
 low    medium   9.2928   5.5426   13.043   True 
 low   very low -2.8915  -8.5801   2.797   False 
medium very low -12.1844 -17.7064 -6.6624   True 
-------------------------------------------------
```

Group "very low" doesn't have significant difference comparing with "low" 

```python
#Test categorical variable urbanrate_freq and numeric variable variable lifeexpectancy
model3 = smf.ols(formula='employrate ~ C(urbanrate_freq)', data=gp)
results3 = model3.fit()
print (results3.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             employrate   R-squared:                       0.090
Model:                            OLS   Adj. R-squared:                  0.075
Method:                 Least Squares   F-statistic:                     5.751
Date:                Sat, 24 Nov 2018   Prob (F-statistic):           0.000897
Time:                        14:24:08   Log-Likelihood:                -662.53
No. Observations:                 178   AIC:                             1333.
Df Residuals:                     174   BIC:                             1346.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                        58.2513      1.620     35.947      0.000      55.053      61.450
C(urbanrate_freq)[T.low]          2.8095      2.153      1.305      0.194      -1.439       7.058
C(urbanrate_freq)[T.medium]      -3.0922      2.044     -1.513      0.132      -7.126       0.942
C(urbanrate_freq)[T.very low]     5.8760      2.698      2.178      0.031       0.550      11.202
==============================================================================
Omnibus:                        1.615   Durbin-Watson:                   1.832
Prob(Omnibus):                  0.446   Jarque-Bera (JB):                1.533
Skew:                          -0.127   Prob(JB):                        0.465
Kurtosis:                       2.623   Cond. No.                         5.29
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

P<0.05 means we are confident enough to reject the hypothesis,which means employ rate are not equal in each urban rate group

```python
#post hoc analysis - Tukey HSDT
mc3 = multi.MultiComparison(in1['employrate'], in1['urbanrate_freq'])
res3 = mc3.tukeyhsd()
print(res3.summary())
```

```
Multiple Comparison of Means - Tukey HSD,FWER=0.05
===============================================
group1  group2  meandiff  lower   upper  reject
-----------------------------------------------
 high    low     2.5033  -2.8662  7.8728 False 
 high   medium  -2.8049  -7.8965  2.2866 False 
 high  very low 11.9529   4.5762 19.3295  True 
 low    medium  -5.3082  -9.9881 -0.6283  True 
 low   very low  9.4496   2.3509 16.5484  True 
medium very low 14.7578   7.8669 21.6488  True 
-----------------------------------------------
```

Group "high" doesn't have significant difference comparing with "medium" and "low"

