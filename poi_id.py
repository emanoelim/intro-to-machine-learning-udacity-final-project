#!/usr/bin/python

import pickle
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from operator import itemgetter

from tester import dump_classifier_and_data, test_classifier


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
def select_clf(n_features, features, labels):

    clf_list = []
    select_k = SelectKBest(f_classif, k=n_features)
    features = select_k.fit_transform(features, labels)
    scores = select_k.scores_
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    dt = DecisionTreeClassifier()
    parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy']}

    clf_dt = GridSearchCV(dt, parameters, scoring='f1')

    rf = RandomForestClassifier()
    parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy'],
                  'n_estimators': [2, 3, 4, 5, 6, 7]}
    clf_rf = GridSearchCV(rf, parameters, scoring='f1')

    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 3, 5, 7, 9]}
    clf_knn = GridSearchCV(knn, parameters, scoring='f1')

    svm = SVC()
    parameters = {'kernel': ['rbf'],
                  'C': [1, 10, 100, 1000, 10000, 100000]}
    clf_svm = GridSearchCV(svm, parameters, scoring='f1')


    clf_dt.fit(features_train, labels_train)
    clf_dt = clf_dt.best_estimator_
    pred_dt = clf_dt.predict(features_test)
    # print '\n\nDecision Tree'
    # print 'Accuracy: ', metrics.accuracy_score(pred_dt, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_dt, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_dt, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_dt, labels_test)
    # print clf_dt
    clf_list.append([metrics.f1_score(pred_dt, labels_test), metrics.accuracy_score(pred_dt, labels_test), n_features, scores, clf_dt])

    clf_rf.fit(features_train, labels_train)
    clf_rf = clf_rf.best_estimator_
    pred_rf = clf_rf.predict(features_test)
    # print '\n\nRandon Forest'
    # print 'Accuracy: ', metrics.accuracy_score(pred_rf, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_rf, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_rf, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_rf, labels_test)
    # print clf_rf
    clf_list.append([metrics.f1_score(pred_rf, labels_test), metrics.accuracy_score(pred_rf, labels_test), n_features, scores, clf_rf])

    clf_knn.fit(features_train, labels_train)
    clf_knn = clf_knn.best_estimator_
    pred_knn = clf_knn.predict(features_test)
    # print '\n\nKNN'
    # print 'Accuracy: ', metrics.accuracy_score(pred_knn, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_knn, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_knn, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_knn, labels_test)
    # print clf_knn
    clf_list.append([metrics.f1_score(pred_knn, labels_test), metrics.accuracy_score(pred_knn, labels_test), n_features, scores, clf_knn])

    clf_svm.fit(features_train, labels_train)
    clf_svm = clf_svm.best_estimator_
    pred_svm = clf_svm.predict(features_test)
    # print '\nSVM'
    # print 'Accuracy: ', metrics.accuracy_score(pred_svm, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_svm, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_svm, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_svm, labels_test)
    # print clf_svm
    clf_list.append([metrics.f1_score(pred_svm, labels_test), metrics.accuracy_score(pred_svm, labels_test), n_features, scores, clf_svm])

    order_clf_list = sorted(clf_list, key=lambda x: x[0])
    return order_clf_list[::-1][0]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = data_dict['METTS MARK'].keys()
features_list.remove('poi')
features_list.remove('email_address')
features_list.remove('total_payments')
features_list.remove('total_stock_value')
features_list.remove('other')


# Remove columns with > 50% NaN's
df = pd.DataFrame(data_dict).T
df.replace(to_replace='NaN', value=np.nan, inplace=True)
for key in features_list:
    if df[key].isnull().sum() > df.shape[0] * 0.5:
        features_list.remove(key)
features_list = ['poi'] + features_list


### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 3: Create new feature(s)
for person in my_dataset:
    if my_dataset[person]['from_this_person_to_poi'] != 'NaN' and my_dataset[person]['to_messages'] != 'NaN':
        percentual_of_messages_sent_to_poi = float(my_dataset[person]['from_this_person_to_poi']) / float(my_dataset[person]['to_messages']) * 100
        percentual_of_messages_received_from_poi = float(my_dataset[person]['from_poi_to_this_person']) / float(my_dataset[person]['from_messages']) * 100
        my_dataset[person]['percentual_of_messages_sent_to_poi'] = percentual_of_messages_sent_to_poi
        my_dataset[person]['percentual_of_messages_received_from_poi'] = percentual_of_messages_received_from_poi
    else:
        my_dataset[person]['percentual_of_messages_sent_to_poi'] = 'NaN'
        my_dataset[person]['percentual_of_messages_received_from_poi'] = 'NaN'
features_list += ['percentual_of_messages_sent_to_poi', 'percentual_of_messages_received_from_poi']
# print features_list


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
clf_list = []
for k in range (1, int(len(features_list) / 2)): # Try sets of 1 - number_of_features / 2
    clf_list.append(select_clf(k, features, labels))
order_clf_list = sorted(clf_list, key=itemgetter(0, 1)) # order by f1-score and accuracy
clf = order_clf_list[len(order_clf_list) - 1][4]

print '\n\nClf: ', clf

number_of_features = order_clf_list[len(order_clf_list) - 1][2]
print '\n\nNumber of features: ', number_of_features

print '\n\nFeatures and scores: '
score_list = order_clf_list[len(order_clf_list) - 1][3]
features = features_list[1:]
features_scores = []
i = 0
for feature in features:
    features_scores.append([feature, score_list[i]])
    i += 1
features_scores = sorted(features_scores, key=itemgetter(1))
print features_scores[::-1]

print '\n\nFeatures used: '
new_features_list = []
for feature in features_scores[::-1][:number_of_features]:
    new_features_list.append(feature[0])
print new_features_list
print '\n\n'
new_features_list = ['poi'] + new_features_list

## Task 6: Dump your classifier, dataset, and features_list so anyone can
## check your results. You do not need to change anything below, but make sure
## that the version of poi_id.py that you submit can be run on its own and
## generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, new_features_list)
test_classifier(clf, my_dataset, new_features_list)
