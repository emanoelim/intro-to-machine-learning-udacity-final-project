# Udacity Intro Machine Learning Final Project

The goal of the final project is to find POIs (persons of interest) in the Enron dataset, studied in the course. 

**Task 1: Select what features you'll use:**
Using human intuition, some of the avaible features were ignored, such as  “email_address”, “other” and features related to the sum of other values (“total_payments” and “total_stock_value”). Features containg more than 50% of NaNs was also ignored. Then, the following list of features remained to be initially used to train the classifier:

- salary;
- to_messages;
- exercised_stock_options;
- bonus;
- restricted_stock;
- shared_receipt_with_poi;
- expenses;
- from_messages;
- from_this_person_to_poi;
- deferred_income;
- from_poi_to_this_person.


**Task 2: Remove outliers:**
Ploting a 2D graph with features “salary” and “bonus” was verified that one person have salary and bonus very larger than the other people. Looking at file enron61702insiderpay.pdf, was verified that these values doesn’t belong to a person, but was correspoding to the sum of all people in the dataset. So, the key “TOTAL” as removed of the data_dict. Looking at the file, we also can see that the last line of the file is labled as “THE TRAVEL AGENCY IN THE PARK”. This key also was removed, since it wasn’t a person.


**Task 3: Create new feature(s):**
Supposing that a Person 1 sends a lot of e-mails (100 e-mails). 50 of these e-mails was sent to a POI. Person 2 sends just a few e-mails (50 e-mails). 40 of these e-mails was sent to a POI. Feature “from_this_person_to_poi” will show a higher value for Person 1, but we know that Person 2 is more suspected of being a POI, since 80% of its e-mails was related to a POI, comparing with Pernson 1 that sent 50% ot its e-mails to a POI.
  
This way, a new feature was created, called “percentual_of_messages_sent_to_poi”, that uses “from_this_person_to_poi” and “to_messages” to compute the percentual of all sent messages that was sent to a POI. The same way, was created a feature called “percentual_of_messages_received_from_poi”, that uses “from_poi_to_this_person” and “from_messages” to compute the percentual of all receveid messages that was sent from a POI.

SelectKBest from scikit-learn was used to show the features scores. The following scores was obteined. It's possible to see that the new features were more relevant than “from_this_person_to_poi”, “to_messages” and “from_messages” used separately.
- exercised_stock_options: 24.250472354526192;
- bonus:  20.25718499812395;
- salary: 17.717873579243289;
- deferred_income:  11.184580251839124;
- restricted_stock:  8.9455030152613304;
- shared_receipt_with_poi:  8.2761382162606445;
- expenses: 5.8153280019048541;
- from_poi_to_this_person: 5.0412573786693846;
- percentual_of_messages_received_from_poi: 4.9530408073262873;
- percentual_of_messages_sent_to_poi:  3.9463436732556301;
- from_this_person_to_poi: 2.2951831957380029;
- to_messages: 1.5425809046549228;
- from_messages: 0.18121500856156128.


**Task 4: Try a varity of classifiers:**
Decision Tree, Random Forest, K-NN and SVM classifiers was used.


**Task 5: Tune your classifier to achieve better than .3 precision and recall:**
GridSearch from scikit-learn was used to tune 4 different types of classifiers (Decision Tree, Random Forest, K-NN and SVM). The following sets of parameters was used: 
```
decision_tree_parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                            'max_features': ['auto', 'sqrt', 'log2', None],
                            'criterion': ['gini', 'entropy']}

randon_forest_parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                            'max_features': ['auto', 'sqrt', 'log2', None],
                            'criterion': ['gini', 'entropy'],
                            'n_estimators': [2, 3, 4, 5, 6, 7]}

knn_parameters = {'n_neighbors': [1, 3, 5, 7, 9]}

svm_parameters = {'kernel': ['rbf'], 
                  'C': [1, 10, 100, 1000, 10000, 100000]}
```
 
Combinations resulting of the GridSearch also was tested with different number of features, 1 – 6 (half of the list features) features. Each group of features was selected using SelectKBest according to higher scores. K-NN classifier using 2 neighbors and 4 features leaded to better results:
- Accuracy:  0.883720930233;
- Recall:  0.6;
- Precision:  0.5;
- F1 score:  0.545454545455.
  
The features selected was: 
- exercised_stock_options;
- bonus;
- salary;
- deferred_income.

Using this classifier with test_classifier(), following results was obtained:
- Accuracy: 0.81521;	
- Precision: 0.37558;	
- Recall: 0.44300;	
- F1: 0.40652;	
- F2: 0.42765;
- Total predictions: 14000;	
- True positives:  886;	
- False positives: 1473;
- False negatives: 1114;	
- True negatives: 10527.

