---
title:  Multiple Regreassion(C3W3)
date: 2018-12-15 00:23:23
categories:
 - Homework -Data
tags: DataAnalysis



---



```python
import numpy as numpyp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn
from scipy import stats


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

```

Center variable


```python

sub1['urbanrate_center'] = (sub1['urbanrate'] - sub1['urbanrate'].mean())
sub1['employrate_center'] = (sub1['employrate'] - sub1['employrate'].mean())
sub1['lifeexpectancy_center'] = (sub1['lifeexpectancy'] - sub1['lifeexpectancy'].mean())
sub1['femaleemployrate_center'] = (sub1['femaleemployrate'] - sub1['femaleemployrate'].mean())
sub1['internetuserate_center'] = (sub1['internetuserate'] - sub1['internetuserate'].mean())

m1 = sub1['urbanrate_center'].mean()
print(m1)
```

    2.200949449793481e-14



```python
m2 = sub1['internetuserate_center'].mean()
print(m2)
```

    -3.3360847959468117e-15


**Try to build a multiple regression model between urban rate, intenet use rate and the income per person**

Look at the distribution of independent and dependent variables


```python
seaborn.distplot(sub1['urbanrate_center'], fit=stats.norm, kde=False)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x11104bd68>




![png](http://img.luhaoip.com/images/2018-12-16-42603.jpg)



```python
seaborn.distplot(sub1['employrate_center'], fit=stats.norm, kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c269998d0>




![png](http://img.luhaoip.com/images/2018-12-16-042601.jpg)



```python
seaborn.distplot(sub1['incomeperperson'], fit=stats.norm, kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c26ac8eb8>




![png](http://img.luhaoip.com/images/2018-12-16-042607.jpg)


Looks like the dependent variable is not a normal distribution

1. Check basic linear regression


```python
scat1 = seaborn.regplot(x="urbanrate_center", y="incomeperperson", scatter=True, data=sub1)
plt.xlabel('Urbanization Rate')
plt.ylabel('Income Per Person')
plt.title ('Scatterplot for the Association Between Urban Rate and Income Per Person')
print(scat1)
```

    AxesSubplot(0.125,0.125;0.775x0.755)



![png](http://img.luhaoip.com/images/2018-12-16-42560.jpg)



```python
print ("OLS regression model for the association between urban rate and Income Per Person")
reg1 = smf.ols('incomeperperson ~ urbanrate_center', data=sub1).fit()
print (reg1.summary())
```

    OLS regression model for the association between urban rate and Income Per Person
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        incomeperperson   R-squared:                       0.380
    Model:                            OLS   Adj. R-squared:                  0.376
    Method:                 Least Squares   F-statistic:                     99.30
    Date:                Sun, 16 Dec 2018   Prob (F-statistic):           1.53e-18
    Time:                        12:23:09   Log-Likelihood:                -1716.9
    No. Observations:                 164   AIC:                             3438.
    Df Residuals:                     162   BIC:                             3444.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept         7692.4504    669.107     11.497      0.000    6371.154    9013.747
    urbanrate_center   289.1789     29.020      9.965      0.000     231.873     346.485
    ==============================================================================
    Omnibus:                       49.867   Durbin-Watson:                   2.097
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               98.863
    Skew:                           1.408   Prob(JB):                     3.41e-22
    Kurtosis:                       5.556   Cond. No.                         23.1
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
fig1=sm.qqplot(reg1.resid, line='r')

```


![png](http://img.luhaoip.com/images/2018-12-16-042606.jpg)


try polynomial to fit the sample better


```python
scat2 = seaborn.regplot(x="urbanrate_center", y="incomeperperson", scatter=True, order = 2, data=sub1)
plt.xlabel('Urbanization Rate')
plt.ylabel('Income Per Person')
plt.title ('Scatterplot for the Association Between Urban Rate and Income Per Person')
print(scat2)
```

    AxesSubplot(0.125,0.125;0.775x0.755)



![png](http://img.luhaoip.com/images/2018-12-16-042602.jpg)



```python
print ("OLS regression model for the association between urban rate and Income Per Person")
reg2 = smf.ols('incomeperperson ~ urbanrate_center+I(urbanrate_center**2)', data=sub1).fit()
print (reg2.summary())
```

    OLS regression model for the association between urban rate and Income Per Person
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        incomeperperson   R-squared:                       0.452
    Model:                            OLS   Adj. R-squared:                  0.445
    Method:                 Least Squares   F-statistic:                     66.36
    Date:                Sun, 16 Dec 2018   Prob (F-statistic):           9.56e-22
    Time:                        12:23:10   Log-Likelihood:                -1706.8
    No. Observations:                 164   AIC:                             3420.
    Df Residuals:                     161   BIC:                             3429.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ============================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------
    Intercept                 4878.1491    879.555      5.546      0.000    3141.197    6615.101
    urbanrate_center           303.1362     27.539     11.007      0.000     248.751     357.521
    I(urbanrate_center ** 2)     5.2938      1.152      4.594      0.000       3.018       7.570
    ==============================================================================
    Omnibus:                       64.147   Durbin-Watson:                   2.091
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              191.103
    Skew:                           1.582   Prob(JB):                     3.18e-42
    Kurtosis:                       7.237   Cond. No.                     1.07e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.07e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig2=sm.qqplot(reg2.resid, line='r')
```


![png](http://img.luhaoip.com/images/2018-12-16-42606.jpg)


The residuals is not normally distributed in both the linear and the polynomial model

2. Try to add another variable (internet use rate) in the multiple regression


```python
print ("OLS regression model for the association between urban rate, internet use rate and Income Per Person")
reg3 = smf.ols('incomeperperson ~ urbanrate_center+I(urbanrate_center**2)+internetuserate_center', data=sub1).fit()
print (reg3.summary())
```

    OLS regression model for the association between urban rate, internet use rate and Income Per Person
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        incomeperperson   R-squared:                       0.702
    Model:                            OLS   Adj. R-squared:                  0.696
    Method:                 Least Squares   F-statistic:                     125.6
    Date:                Sun, 16 Dec 2018   Prob (F-statistic):           7.58e-42
    Time:                        12:23:10   Log-Likelihood:                -1656.8
    No. Observations:                 164   AIC:                             3322.
    Df Residuals:                     160   BIC:                             3334.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ============================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------
    Intercept                 5373.7664    652.067      8.241      0.000    4085.997    6661.536
    urbanrate_center            79.4911     28.067      2.832      0.005      24.062     134.921
    I(urbanrate_center ** 2)     4.3615      0.856      5.093      0.000       2.670       6.053
    internetuserate_center     268.0078     23.135     11.584      0.000     222.318     313.698
    ==============================================================================
    Omnibus:                       37.581   Durbin-Watson:                   2.206
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               88.539
    Skew:                           0.971   Prob(JB):                     5.94e-20
    Kurtosis:                       6.030   Cond. No.                     1.07e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.07e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig3=sm.qqplot(reg3.resid, line='r')
```


![png](http://img.luhaoip.com/images/2018-12-16-042605.jpg)

R-square became larger and the p-value < 0.05, but the residuals are still not noramlly distributed.

*No significant correlation relationship changed after adding internet use rate, so there's no potential confounder.*

**Try to transform dependent variable to fit the model better**


```python
sub1['incomeperperson_log'] = numpyp.log(sub1['incomeperperson'])
seaborn.distplot(sub1['incomeperperson_log'], fit=stats.norm, kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c27486dd8>




![png](http://img.luhaoip.com/images/2018-12-16-42609.jpg)



```python
print ("OLS regression model for the association between urban rate plus internet use rate and Transformed Income Per Person")
reg4 = smf.ols('incomeperperson_log ~ urbanrate_center+I(urbanrate_center**2)+internetuserate_center', data=sub1).fit()
print (reg4.summary())
```

    OLS regression model for the association between urban rate plus internet use rate and Transformed Income Per Person
                                 OLS Regression Results                            
    ===============================================================================
    Dep. Variable:     incomeperperson_log   R-squared:                       0.809
    Model:                             OLS   Adj. R-squared:                  0.805
    Method:                  Least Squares   F-statistic:                     225.7
    Date:                 Sun, 16 Dec 2018   Prob (F-statistic):           2.96e-57
    Time:                         12:23:11   Log-Likelihood:                -174.08
    No. Observations:                  164   AIC:                             356.2
    Df Residuals:                      160   BIC:                             368.6
    Df Model:                            3                                         
    Covariance Type:             nonrobust                                         
    ============================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------
    Intercept                    7.8241      0.077    101.309      0.000       7.672       7.977
    urbanrate_center             0.0257      0.003      7.736      0.000       0.019       0.032
    I(urbanrate_center ** 2)  8.392e-05      0.000      0.827      0.409      -0.000       0.000
    internetuserate_center       0.0348      0.003     12.708      0.000       0.029       0.040
    ==============================================================================
    Omnibus:                        9.131   Durbin-Watson:                   2.156
    Prob(Omnibus):                  0.010   Jarque-Bera (JB):                9.493
    Skew:                           0.471   Prob(JB):                      0.00868
    Kurtosis:                       3.709   Cond. No.                     1.07e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.07e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.


urbanrate_center^2's p-value is 0.409, it is not significant in the new model. Delete this explanatory variable.


```python
print ("OLS regression model for the association between urban rate plus internet use rate and Transformed Income Per Person")
reg5 = smf.ols('incomeperperson_log ~ urbanrate_center+internetuserate_center', data=sub1).fit()
print (reg5.summary())
```

    OLS regression model for the association between urban rate plus internet use rate and Transformed Income Per Person
                                 OLS Regression Results                            
    ===============================================================================
    Dep. Variable:     incomeperperson_log   R-squared:                       0.808
    Model:                             OLS   Adj. R-squared:                  0.806
    Method:                  Least Squares   F-statistic:                     338.8
    Date:                 Sun, 16 Dec 2018   Prob (F-statistic):           2.00e-58
    Time:                         12:23:11   Log-Likelihood:                -174.43
    No. Observations:                  164   AIC:                             354.9
    Df Residuals:                      161   BIC:                             364.2
    Df Model:                            2                                         
    Covariance Type:             nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                  7.8687      0.055    142.445      0.000       7.760       7.978
    urbanrate_center           0.0253      0.003      7.704      0.000       0.019       0.032
    internetuserate_center     0.0350      0.003     12.856      0.000       0.030       0.040
    ==============================================================================
    Omnibus:                        8.744   Durbin-Watson:                   2.176
    Prob(Omnibus):                  0.013   Jarque-Bera (JB):                9.210
    Skew:                           0.444   Prob(JB):                       0.0100
    Kurtosis:                       3.748   Cond. No.                         33.3
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The new model's R-square is 0.808 and the p-value is small.


```python
#residual qq-plot
fig5=sm.qqplot(reg5.resid, line='r')
```


![png](http://img.luhaoip.com/images/2018-12-16-042600.jpg)


The residuals distribution became a lot of better


```python

# simple plot of residuals
stdres=pd.DataFrame(reg5.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')

```




    Text(0.5, 0, 'Observation Number')




![png](http://img.luhaoip.com/images/2018-12-16-042604.jpg)



```python
fig = plt.figure(figsize=(12,8))
fig5_1 = sm.graphics.plot_regress_exog(reg5,  "urbanrate_center", fig=fig)
print(fig5_1)
```

    Figure(864x576)



![png](http://img.luhaoip.com/images/2018-12-16-042603.jpg)



```python
fig = plt.figure(figsize=(12,8))
fig5_2 = sm.graphics.plot_regress_exog(reg5,  "urbanrate_center", fig=fig)
print(fig5_2)
```

    Figure(864x576)



![png](http://img.luhaoip.com/images/2018-12-16-042608.jpg)



```python
# leverage plot
fig5=sm.graphics.influence_plot(reg5, size=8)
print(fig5)
```

    Figure(432x288)



![png](http://img.luhaoip.com/images/2018-12-16-042559.jpg)


**Summary**

The new model (transformed income per person as dependent variable, urban rate and internet use rate as independent variables) is a more ideal one. And it could pass the model fit evaluation.

