---
title:  Decision Tree(C4W1)
date: 2018-12-19  00:23:23
categories:
 - Homework -Data
tags: DataAnalysis
---



```python

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics


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


```python

#dependent variable tranformation
def HIINCOME (x):
   if x['incomeperperson']>= 12235:
      return 1
   else: 
      return 0
  
sub1['HIINCOME'] = sub1.apply (lambda x: HIINCOME (x), axis=1)
```

Build the model


```python

predictors = sub1[['urbanrate', 'internetuserate','employrate', 'femaleemployrate','lifeexpectancy']]
targets = sub1.HIINCOME

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

```


```python
pred_train.shape
```




    (98, 5)




```python
pred_test.shape
```




    (66, 5)




```python
tar_train.shape
```




    (98,)




```python
tar_test.shape
```




    (66,)




```python
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
```




    array([[55,  2],
           [ 4,  5]])



55 true positive and 2 true negative prediction. 4 false positive and 5 false negative prediction.


```python
sklearn.metrics.accuracy_score(tar_test, predictions)

```




    0.9090909090909091



Accuraty score is 0.909


```python

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
```




![png](http://img.luhaoip.com/images/2018-12-19-224911.jpg)



Respond variable is a binary variable - High income, which came from the income per person of Gapminder dataset and it was transfered to binary based on the definitio of high income by World bank （https://en.wikipedia.org/wiki/World_Bank_high-income_economy。 

The following explanatory variables were included as possible contributors to a classification tree model: Urban rate(X0), internet use rate(X1), employ rate(X2), female employ rate(X3) and life expectancy(X4)

The decision tree analysis was performed to test the non linear relationship among a series of explanatory variables and a binary, categorical response variable.

The life expectancy was the first variable to separate the sample. In the subgroup life expectancy <= 77.663, 69 countries are not high income, 1 country is high income country. The other group with life epectancy > 77.663, 3 are not high income countries while 25 are high income countries.

Of the life expectancy lower thatn 77.663, the only high income coutry are subdivided into one group with internet use rate > 76, and the rest of the group are all lower than 76. In the other group with life expectancy greater than 77.663, 88% of high income countries(22/25) are having internet use rate greater than 64.4.

We could say life expectancy and internet use rate are the two most imporant explanatory variables to predict income level. Further considering other variables may cause a overfitting since the sample is pretty small.
