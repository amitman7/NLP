import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from stylometry.extract import *
import stylometry

"""

# Specify the file paths
file_paths_old = glob.glob('stylometry-data/BERT/Dor Ha-Medina BERT output/*.json')
file_paths_new = glob.glob('stylometry-data/BERT/Present BERT output/*.json')


# extracting the Syntaxes
# old songs 
for file_path in file_paths_old:
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)

    string_pos = ''
    for i in range (len(data['tokens'])):
        string_pos = string_pos + " " + (data['tokens'][i]['morph']['pos'])    

    file_path = file_path[:-10]
    file_path = file_path + '.txt'
    with open(file_path, 'w') as file:
        file.write(string_pos)

# new songs read
for file_path in file_paths_new:
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)

    string_pos = ''
    for i in range (len(data['tokens'])):
        string_pos = string_pos + " " + (data['tokens'][i]['morph']['pos'])    

    file_path = file_path[:-10]
    file_path = file_path + '.txt'
    with open(file_path, 'w') as file:
        file.write(string_pos)


 """    

# old data - feature extraction
old_data = StyloCorpus.from_glob_pattern('stylometry-data/BERT/old/*.txt')
old_data.output_csv('stylometry-data/BERT/old_ALL.csv')

# new data - feature extraction
new_data = StyloCorpus.from_glob_pattern('stylometry-data/BERT/new/*.txt')
new_data.output_csv('stylometry-data/BERT/new_ALL.csv')



# Load data from CSV file old.csv
old_data = pd.read_csv("stylometry-data/BERT/old_ALL.csv",encoding="windows-1255", index_col=False)
sum_rows_old = len(old_data)
value_old = []
for i in range (sum_rows_old): 
    value_old.append(-1)
old_data['label'] = value_old


# Load data from CSV file new.csv
new_data = pd.read_csv("stylometry-data/BERT/new_ALL.csv",encoding="windows-1255",index_col=False)
sum_rows_new = len(new_data)
value_new = []
for i in range (sum_rows_new): 
    value_new.append(1)
new_data['label'] = value_new



# Prepare data
# removing the first two cols - Author and Title
old_data = old_data.iloc[:, 2:]
new_data = new_data.iloc[:, 2:]

#last col has the label 
X_old = old_data.iloc[:, :-1]
y_old = old_data.iloc[:, -1]

X_new = new_data.iloc[:, :-1]
y_new = new_data.iloc[:, -1]


# Combine data
X = pd.concat([X_old, X_new])
y = pd.concat([y_old, y_new])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model_LogisticRegression = LogisticRegression(max_iter=1000)
model_LogisticRegression.fit(X_train, y_train)

# Train svm model
model_svm_liblinear = LinearSVC()
model_svm_liblinear.fit(X_train, y_train)

# Train svm_libsvm model
model_svm_libsvm = SVC()
model_svm_libsvm.fit(X_train, y_train)

# Make predictions
y_pred_LogisticRegression = model_LogisticRegression.predict(X_test)
y_pred_svm_liblinear = model_svm_liblinear.predict(X_test)
y_pred_svm_libsvm = model_svm_libsvm.predict(X_test)


# Evaluate model LogisticRegression
accuracy = accuracy_score(y_test, y_pred_LogisticRegression)
precision = precision_score(y_test, y_pred_LogisticRegression)
recall = recall_score(y_test, y_pred_LogisticRegression)
f1 = f1_score(y_test, y_pred_LogisticRegression)
conf_matrix = confusion_matrix(y_test, y_pred_LogisticRegression)

print("LogisticRegression model:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix,"\n")


# Evaluate model svm_liblinear
accuracy = accuracy_score(y_test, y_pred_svm_liblinear)
precision = precision_score(y_test, y_pred_svm_liblinear)
recall = recall_score(y_test, y_pred_svm_liblinear)
f1 = f1_score(y_test, y_pred_svm_liblinear)
conf_matrix = confusion_matrix(y_test, y_pred_svm_liblinear)

print("svm_liblinear model:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix,"\n")



# Evaluate model svm_libsvm
accuracy = accuracy_score(y_test, y_pred_svm_libsvm)
precision = precision_score(y_test, y_pred_svm_libsvm)
recall = recall_score(y_test, y_pred_svm_libsvm)
f1 = f1_score(y_test, y_pred_svm_libsvm)
conf_matrix = confusion_matrix(y_test, y_pred_svm_libsvm)

print("svm_libsvm model:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix,"\n")

