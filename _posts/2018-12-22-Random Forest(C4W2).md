---
title:  Random Forest (C4W2)
date: 2018-12-22  00:23:23
categories:
 - Homework -Data
tags: DataAnalysis

---



```python
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier


# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pd.read_csv('gapminder.csv')

data.dtypes

```




    country                 object
    incomeperperson         object
    alcconsumption          object
    armedforcesrate         object
    breastcancerper100th    object
    co2emissions            object
    femaleemployrate        object
    hivrate                 object
    internetuserate         object
    lifeexpectancy          object
    oilperperson            object
    polityscore             object
    relectricperperson      object
    suicideper100th         object
    employrate              object
    urbanrate               object
    dtype: object




```python
data.describe()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>incomeperperson</th>
      <th>alcconsumption</th>
      <th>armedforcesrate</th>
      <th>breastcancerper100th</th>
      <th>co2emissions</th>
      <th>femaleemployrate</th>
      <th>hivrate</th>
      <th>internetuserate</th>
      <th>lifeexpectancy</th>
      <th>oilperperson</th>
      <th>polityscore</th>
      <th>relectricperperson</th>
      <th>suicideper100th</th>
      <th>employrate</th>
      <th>urbanrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
      <td>213</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>213</td>
      <td>191</td>
      <td>181</td>
      <td>165</td>
      <td>137</td>
      <td>201</td>
      <td>154</td>
      <td>47</td>
      <td>193</td>
      <td>190</td>
      <td>64</td>
      <td>22</td>
      <td>133</td>
      <td>192</td>
      <td>140</td>
      <td>195</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Liberia</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>23</td>
      <td>26</td>
      <td>49</td>
      <td>40</td>
      <td>13</td>
      <td>35</td>
      <td>66</td>
      <td>21</td>
      <td>22</td>
      <td>150</td>
      <td>52</td>
      <td>77</td>
      <td>22</td>
      <td>35</td>
      <td>10</td>
    </tr>
  </tbody>
</table>





```python

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


#dependent variable tranformation
def HIINCOME (x):
   if x['incomeperperson']>= 12235:
      return 1
   else: 
      return 0
  
data['HIINCOME'] = data.apply (lambda x: HIINCOME (x), axis=1)

data_clean = data.dropna()

data_clean.dtypes
```




    country                  object
    incomeperperson         float64
    alcconsumption          float64
    armedforcesrate         float64
    breastcancerper100th    float64
    co2emissions            float64
    femaleemployrate        float64
    hivrate                  object
    internetuserate         float64
    lifeexpectancy          float64
    oilperperson             object
    polityscore             float64
    relectricperperson      float64
    suicideper100th         float64
    employrate              float64
    urbanrate               float64
    HIINCOME                  int64
    dtype: object




```python
data_clean.describe()

```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>incomeperperson</th>
      <th>alcconsumption</th>
      <th>armedforcesrate</th>
      <th>breastcancerper100th</th>
      <th>co2emissions</th>
      <th>femaleemployrate</th>
      <th>internetuserate</th>
      <th>lifeexpectancy</th>
      <th>polityscore</th>
      <th>relectricperperson</th>
      <th>suicideper100th</th>
      <th>employrate</th>
      <th>urbanrate</th>
      <th>HIINCOME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
      <td>121.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8079.87</td>
      <td>7.39</td>
      <td>1.51</td>
      <td>40.35</td>
      <td>8173718515.15</td>
      <td>46.06</td>
      <td>38.11</td>
      <td>71.38</td>
      <td>4.31</td>
      <td>1108.81</td>
      <td>9.83</td>
      <td>57.74</td>
      <td>60.92</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10518.56</td>
      <td>5.25</td>
      <td>1.62</td>
      <td>23.60</td>
      <td>32754566375.00</td>
      <td>13.57</td>
      <td>27.48</td>
      <td>8.70</td>
      <td>6.34</td>
      <td>1618.50</td>
      <td>6.38</td>
      <td>9.27</td>
      <td>20.10</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>min</th>
      <td>103.78</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>3.90</td>
      <td>7315000.00</td>
      <td>12.40</td>
      <td>0.72</td>
      <td>48.40</td>
      <td>-10.00</td>
      <td>0.00</td>
      <td>0.20</td>
      <td>37.40</td>
      <td>15.10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1036.83</td>
      <td>3.11</td>
      <td>0.53</td>
      <td>23.40</td>
      <td>188268666.67</td>
      <td>39.20</td>
      <td>12.33</td>
      <td>67.71</td>
      <td>0.00</td>
      <td>213.06</td>
      <td>4.96</td>
      <td>51.20</td>
      <td>48.36</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2923.14</td>
      <td>6.66</td>
      <td>1.01</td>
      <td>33.00</td>
      <td>999874333.33</td>
      <td>47.50</td>
      <td>35.85</td>
      <td>73.49</td>
      <td>7.00</td>
      <td>537.10</td>
      <td>8.97</td>
      <td>58.20</td>
      <td>63.30</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10749.42</td>
      <td>11.10</td>
      <td>1.81</td>
      <td>51.60</td>
      <td>4200940333.33</td>
      <td>53.80</td>
      <td>62.47</td>
      <td>77.69</td>
      <td>9.00</td>
      <td>1411.23</td>
      <td>13.09</td>
      <td>63.10</td>
      <td>74.50</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>39972.35</td>
      <td>23.01</td>
      <td>9.82</td>
      <td>101.10</td>
      <td>334220872333.33</td>
      <td>82.20</td>
      <td>93.28</td>
      <td>83.39</td>
      <td>10.00</td>
      <td>11154.76</td>
      <td>33.34</td>
      <td>80.70</td>
      <td>100.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>





```python

predictors = data_clean[['alcconsumption','armedforcesrate','breastcancerper100th','co2emissions','polityscore','relectricperperson','suicideper100th','urbanrate', 'internetuserate','employrate', 'femaleemployrate','lifeexpectancy']]

targets = data_clean.HIINCOME


pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4)

```


```python
pred_train.shape

```




    (72, 12)




```python
pred_test.shape
```




    (49, 12)




```python
tar_train.shape
```




    (72,)




```python
tar_test.shape
```




    (49,)




```python

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)

```




    array([[35,  0],
           [ 4, 10]])




```python
sklearn.metrics.accuracy_score(tar_test, predictions)

```




    0.9183673469387755




```python

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)
```

    [0.02848068 0.01130221 0.11655563 0.05327637 0.03305181 0.13373736
     0.02599521 0.01971851 0.39190565 0.02868403 0.01260359 0.14468895]





![png](http://img.luhaoip.com/images/2018-12-22-090007.jpg)



After running the model for a couple times, the feature importance is not very stable. From the output we could tell the most important varibales are residential electricity consumption(relectricperperon), per person， life expectancy, internet use rate and breast cancer per 100 person. The optimal number of trees is around 5. 

Due the the gapminder data set is relatively small, the model outcome vary pretty much. The random forst is useful to identify the important variables. However, it not very reproducible as a predictive model.

