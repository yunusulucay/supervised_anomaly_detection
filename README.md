# SUPERVISED ANOMALY DETECTION

Anomaly detection has 3 parts. In this project I examined supervised learning anomaly detection. This is actually a classification problem. But there are some difference like used metrics(mahalanobis distance instead of eucledian distance)[1]. Or unbalanced data. But because of this dataset has a balanced dataset, I didn't touch those problems. There are 3 steps in this project: exploratory data analysis, feature engineering and modelling. I implemented 9 models(Logistic Regression, Stochastic Gradient Descent Classifier, Passive-Aggressive Algorithms, LightGBM, Extra Trees, Neural Networks, KNN, Naive Bayes and XGBOD(XGBClassifier from PyOD))

![image](https://user-images.githubusercontent.com/42489236/153754116-36893572-3cfe-4f38-950b-921c9d640e32.png)
Difference Between Eucledian Distance and Mahalanobis Distance Depending on Correlation[1]

Depending on models' comparison winner is XGBClassifier.

**Imbalanced classification. Binary Classification. Sources.**

- https://towardsdatascience.com/supervised-machine-learning-technique-for-anomaly-detection-logistic-regression-97fc7a9cacd4

- https://stats.stackexchange.com/questions/474640/is-anomaly-detection-supervised-or-un-supervised

- https://github.com/yzhao062/pyod

- https://www.analyticsvidhya.com/blog/2021/06/univariate-anomaly-detection-a-walkthrough-in-python/

- https://www.projectpro.io/article/anomaly-detection-using-machine-learning-in-python-with-example/555

- https://medium.com/learningdatascience/anomaly-detection-techniques-in-python-50f650c75aaf

- https://www.machinelearningplus.com/statistics/mahalanobis-distance/

**If you have an imbalanced data(like 90000 labeled with 1 and 100 data labeled 0) you can look the links below.**

- https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

- https://www.kaggle.com/klaudiajankowska/binary-classification-multiple-method-comparison

- https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

- https://www.kaggle.com/satoshiss/titanic-binary-classification
