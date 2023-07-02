# imports relevant libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# importing Data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data (only if needed)
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (if needed)
sc_X = StandardScaler()
standard_X = sc_X.fit_transform(X_train)
standard_testX = sc_X.transform(X_test)

# Training the Regression models on the Training set

# 1. Logistic Regression
logistic_regression_model = LogisticRegression(random_state=0)
logistic_regression_model.fit(standard_X, y_train)
# 2. KNN
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(standard_X, y_train)
# 3. Linear SVM
linear_svm_model = SVC(kernel ='linear', random_state = 0)
linear_svm_model.fit(standard_X, y_train)
# 4. Kernel SVM
kernel_svm_model = SVC(kernel ='rbf', random_state = 0)
kernel_svm_model.fit(standard_X, y_train)
# 5. Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(standard_X, y_train)
# 6. Decision Tree Classification
decisionTree_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
decisionTree_model.fit(standard_X, y_train)
# 7. Random Forest Classification
randomForest_model = RandomForestClassifier(n_estimators=10, random_state=0)
randomForest_model.fit(standard_X, y_train)

# Predicting the Test set results
# 1. Logistic Regression
logistic_pred = logistic_regression_model.predict(standard_testX)
# 2. KNN
knn_pred = knn_model.predict(standard_testX)
# 3. Linear SVM
linear_svm_pred = linear_svm_model.predict(standard_testX)
# 4. Kernel SVM
kernel_svm_pred = kernel_svm_model.predict(standard_testX)
# 5. Naive Bayes
naive_bayes_pred = naive_bayes_model.predict(standard_testX)
# 6. Decision Tree Classification
dtc_pred = decisionTree_model.predict(standard_testX)
# 7. Random Forest Classification
rfc_pred = randomForest_model.predict(standard_testX)

# Evaluating the Model Performance
# 1. Logistic Regression
logistic_score = accuracy_score(y_test, logistic_pred)
logistic_cm = confusion_matrix(y_test, logistic_pred)
# 2. KNN
knn_score = accuracy_score(y_test, knn_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
# 3. Linear SVM
lsvm_score = accuracy_score(y_test, linear_svm_pred)
lsvm_cm = confusion_matrix(y_test, linear_svm_pred)
# 4. Kernel SVM
ksvm_score = accuracy_score(y_test, kernel_svm_pred)
ksvm_cm = confusion_matrix(y_test, kernel_svm_pred)
# 5. Naive Bayes
naive_score = accuracy_score(y_test, naive_bayes_pred)
naive_cm = confusion_matrix(y_test, naive_bayes_pred)
# 6. Decision Tree Classification
dtc_score = accuracy_score(y_test, dtc_pred)
dtc_cm = confusion_matrix(y_test, dtc_pred)
# 7. Random Forest Classification
rfc_score = accuracy_score(y_test, rfc_pred)
rfc_cm = confusion_matrix(y_test, rfc_pred)

results = {
    "Logistic Regression": logistic_score,
    "KNN": knn_score,
    "Linear SVM": lsvm_score,
    "Kernel SVM": ksvm_score,
    "Naive Bayes": naive_score,
    "Decision Tree Classification": dtc_score,
    "Random Forest Classification": rfc_score
}

results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i in range(0, len(results)):
    print(f'The model in the {i + 1} place is {results[i][0]} and the accuracy score is {results[i][1]}\n')
