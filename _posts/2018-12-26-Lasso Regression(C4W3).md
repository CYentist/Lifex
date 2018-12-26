---
title:  Lasso Regression (C4W3)
date: 2018-12-26  00:23:23
categories:

 - Homework -Data
tags: DataAnalysis

---


```python

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV


data = pd.read_csv('gapminder.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)

# convert variables to numeric format using convert_objects function
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['femaleemployrate'] = pd.to_numeric(data['femaleemployrate'], errors='coerce')
data['internetuserate'] = pd.to_numeric(data['internetuserate'], errors='coerce')
data['alcconsumption'] = pd.to_numeric(data['alcconsumption'], errors='coerce')
data['armedforcesrate'] = pd.to_numeric(data['armedforcesrate'], errors='coerce')
data['breastcancerper100th'] = pd.to_numeric(data['breastcancerper100th'], errors='coerce')
data['co2emissions'] = pd.to_numeric(data['co2emissions'], errors='coerce')
data['polityscore'] = pd.to_numeric(data['polityscore'], errors='coerce')
data['relectricperperson'] = pd.to_numeric(data['relectricperperson'], errors='coerce')
data['suicideper100th'] = pd.to_numeric(data['suicideper100th'], errors='coerce')

data_clean = data.dropna()

predictors = data_clean[['alcconsumption','armedforcesrate','breastcancerper100th','co2emissions','polityscore','relectricperperson','suicideper100th','urbanrate', 'internetuserate','employrate', 'femaleemployrate','lifeexpectancy']]

targets = data_clean['incomeperperson']

predictors_p=predictors.copy()

from sklearn import preprocessing
predictors_p['employrate']=preprocessing.scale(predictors_p['employrate'].astype('float64'))
predictors_p['lifeexpectancy']=preprocessing.scale(predictors_p['lifeexpectancy'].astype('float64'))
predictors_p['urbanrate']=preprocessing.scale(predictors_p['urbanrate'].astype('float64'))
predictors_p['employrate']=preprocessing.scale(predictors_p['employrate'].astype('float64'))
predictors_p['femaleemployrate']=preprocessing.scale(predictors_p['femaleemployrate'].astype('float64'))
predictors_p['internetuserate']=preprocessing.scale(predictors_p['internetuserate'].astype('float64'))
predictors_p['alcconsumption']=preprocessing.scale(predictors_p['alcconsumption'].astype('float64'))
predictors_p['armedforcesrate']=preprocessing.scale(predictors_p['armedforcesrate'].astype('float64'))
predictors_p['breastcancerper100th']=preprocessing.scale(predictors_p['breastcancerper100th'].astype('float64'))
predictors_p['polityscore']=preprocessing.scale(predictors_p['polityscore'].astype('float64'))
predictors_p['relectricperperson']=preprocessing.scale(predictors_p['relectricperperson'].astype('float64'))
predictors_p['suicideper100th']=preprocessing.scale(predictors_p['suicideper100th'].astype('float64'))

```

Using gapminder data, I choose Incomer pre person as the target variable, then put the rest of variables into the rasso regression model. Try to find the most important variables using lasso regression.


```python

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors_p, targets, 
                                                              test_size=.3, random_state=123)
#CV = 10 means 10 folds Cross Validation Process
#precompute=False tells Python not to use a precomputed matrix(This coudl speed up when dealing with large datasets)
# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
```


```python

#The dict object creates a dictionary, and the zip object creates lists.
# print variable names and regression coefficients
dict(zip(predictors_p.columns, model.coef_))


```




    {'alcconsumption': -1094.9679042556913,
     'armedforcesrate': 0.0,
     'breastcancerper100th': 2099.7103302351034,
     'co2emissions': 5.179112129737011e-08,
     'polityscore': 220.53594441281524,
     'relectricperperson': 6121.2581219209815,
     'suicideper100th': 0.0,
     'urbanrate': 336.5656316724012,
     'internetuserate': 3697.3664621768175,
     'employrate': 861.4061128900834,
     'femaleemployrate': 0.0,
     'lifeexpectancy': 0.0}



Accoding to the coefficients list, the most important variables to predict incomer per person are residential electricity consumption(relectricperperon) per person, internet use rate and breast cancer per 100 person. Comparing to the result of random forest from the previous week, which result is residential electricity consumption(relectricperperon) per person， life expectancy, internet use rate and breast cancer per 100 person, the only differrence is the coefficient of life expectancy was shrinked to zero. 

The reason of behind may be life expectancy is a confounder of brease cancer rate.


```python

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
```




    Text(0.5,1,'Regression Coefficients Progression for Lasso Paths')




![png](http://img.luhaoip.com/images/2018-12-26-075107.jpg)



```python


# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         
```

    Text(0.5,1,'Mean squared error on each fold')




![png](http://img.luhaoip.com/images/2018-12-26-075103.jpg)


One intresting finding is the mean squared error increased when the alpha is lower than 2.5, and the regression coeffienct plot indicated that when alpha = 2.5, only 3 variables's coefficient was not shrinked to zero, which is residential electricity consumption(relectricperperon) per person, internet use rate and breast cancer per 100 person. This could be interpreted as the 3 variables is good enough to predict the target variable income per person.


```python

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

```

    training data MSE
    20491598.169149857
    test data MSE
    62222181.64197244



```python

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
```

    training data R-square
    0.8145791630282075
    test data R-square
    0.41762666661965286


The model could predict 41.7% of testing data set, which is pretty high. However, like I just commented, life expectancy and breast cancer rate may have corelation between each other, so this may influcence the model.

