# Udacity Intro Machine Learning Final Project

The goal of the final project is to find POIs (persons of interest) in the Enron dataset, studied in the course. 

Task 1: Select what features you'll use:
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

Task 2: Remove outliers:
Ploting a 2D graph with features “salary” and “bonus” was verified that one person have salary and bonus very larger than the other people. Looking at file enron61702insiderpay.pdf, was verified that these values doesn’t belong to a person, but was correspoding to the sum of all people in the dataset. So, the key “TOTAL” as removed of the data_dict. Looking at the file, we also can see that the last line of the file is labled as “THE TRAVEL AGENCY IN THE PARK”. This key also was removed, since it wasn’t a person.
