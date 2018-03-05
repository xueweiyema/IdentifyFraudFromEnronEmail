#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list=['poi','bonus','exercised_stock_options','expenses','other','restricted_stock','salary','shared_receipt_with_poi','total_payments','total_stock_value','from_ratio','to_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_df = pd.DataFrame(data_dict)
person_df=data_df.transpose()
person_df.replace('NaN',np.nan,inplace=True)


### Task 2: Remove outliers

person_df.drop(['TOTAL'],inplace=True)
### Task 3: Create new feature(s)

person_df.drop(['email_address','deferral_payments','deferred_income','director_fees','loan_advances','long_term_incentive','restricted_stock_deferred'],axis=1, inplace=True)
temp_df=person_df[['from_this_person_to_poi','from_messages','from_poi_to_this_person','to_messages']].dropna(how='any')
person_df['from_ratio']=temp_df['from_this_person_to_poi']/temp_df['from_messages']
person_df['to_ratio']=temp_df['from_poi_to_this_person']/temp_df['to_messages']

# 
person_df['from_ratio'].fillna(person_df['from_ratio'].mean(),inplace=True)
person_df['to_ratio'].fillna(person_df['to_ratio'].mean(),inplace=True)

# 
person_df.drop(['from_this_person_to_poi','from_messages','from_poi_to_this_person','to_messages'],axis=1, inplace=True)

person_df['bonus']=np.log1p(person_df['bonus'])
person_df['bonus'].fillna(person_df['bonus'].mean(),inplace=True)

person_df['exercised_stock_options']=np.log1p(person_df['exercised_stock_options'])
person_df['exercised_stock_options'].fillna(person_df['exercised_stock_options'].mean(),inplace=True)

person_df['expenses']=np.log1p(person_df['expenses'])
person_df['expenses'].fillna(person_df['expenses'].mean(),inplace=True)

person_df['other']=np.log1p(person_df['other'])
person_df['other'].fillna(person_df['other'].mean(),inplace=True)

#
person_df.loc[person_df.index=='BHATNAGAR SANJAY','restricted_stock']=abs(person_df['restricted_stock']['BHATNAGAR SANJAY'])

person_df['restricted_stock']=np.log1p(person_df['restricted_stock'])
person_df['restricted_stock'].fillna(person_df['restricted_stock'].mean(),inplace=True)

person_df['salary']=np.log1p(person_df['salary'])
person_df['salary'].fillna(person_df['salary'].mean(),inplace=True)

person_df['shared_receipt_with_poi']=np.log1p(person_df['shared_receipt_with_poi'])
person_df['shared_receipt_with_poi'].fillna(person_df['shared_receipt_with_poi'].mean(),inplace=True)

person_df['total_payments']=np.log1p(person_df['total_payments'])
person_df['total_payments'].fillna(person_df['total_payments'].mean(),inplace=True)

#
person_df.loc[person_df.index=='BELFER ROBERT','total_stock_value']=abs(person_df['total_stock_value']['BELFER ROBERT'])

person_df['total_stock_value']=np.log1p(person_df['total_stock_value'])
person_df['total_stock_value'].fillna(person_df['total_stock_value'].mean(),inplace=True)





### Store to my_dataset for easy export below.
my_dataset = person_df.to_dict(orient ='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.8)

regressor = DecisionTreeClassifier()
parameters = {'min_samples_leaf':[1,2,3,4,5,6], 'min_samples_split':[2,3,4,5]}
# scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)

grid = GridSearchCV(regressor, parameters, scoring='f1', cv=kfold)
grid = grid.fit(X_train, y_train)
reg = grid.best_estimator_

for key in parameters.keys():
    print('%s: %d'%(key, reg.get_params()[key]))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dtclf=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
test_classifier(dtclf, person_df.to_dict(orient ='index'), features_list)
dump_classifier_and_data(clf, my_dataset, features_list)