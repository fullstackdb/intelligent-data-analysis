# learn more https://en.wikipedia.org/wiki/Support_vector_machine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# using absolute path to prevent WORKDIR issue
bankdata = pd.read_csv('/Users/azhmakin/Documents/projects/own/university/intelligent-data-analysis/lab-4/assets/bill_authentication.csv');

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

# Data Preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Training the Algorithm
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
