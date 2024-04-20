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
import matplotlib.pyplot as plt
import seaborn as sns



# old - feature extraction
old_data = StyloCorpus.from_glob_pattern('stylometry-data/hebrew/old/*.txt')
old_data.output_csv('stylometry-data/hebrew/old_ALL.csv')

# new - feature extraction
new_data = StyloCorpus.from_glob_pattern('stylometry-data/hebrew/new/*.txt')
new_data.output_csv('stylometry-data/hebrew/new_ALL.csv')

# Load data from CSV file old.csv
old_data = pd.read_csv('stylometry-data/hebrew/old_ALL.csv',encoding="windows-1255")
# removing last col of the file - contains garbege cause of the hebrew
old_data = old_data.iloc[:, :-1]

sum_rows_old = len(old_data)
value_old = []
for i in range (sum_rows_old): 
    value_old.append(-1)
old_data['label'] = value_old


# Load data from CSV file new.csv
new_data = pd.read_csv('stylometry-data/hebrew/new_ALL.csv',encoding="windows-1255")
# removing last col of the file - contains garbege cause of the hebrew
new_data = new_data.iloc[:, :-1]

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

model_svm_liblinear = LinearSVC()
model_svm_liblinear.fit(X_train, y_train)

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


# new data prediction
s = StyloCorpus.from_glob_pattern('stylometry-data/prediction_check/*.txt')
s.output_csv('stylometry-data/prediction_check/prediction_check.csv')

new_data = pd.read_csv("stylometry-data/prediction_check/prediction_check.csv",encoding="windows-1255")
song_name = new_data["Title"]
author_name = new_data["Author"]
data_sum = len (song_name)

new_data = new_data.iloc[:, 2:]
new_data = new_data.iloc[:, :-1]

new_data_predictions_LogisticRegression = model_LogisticRegression.predict(new_data)
new_data_predictions_liblinear = model_svm_liblinear.predict(new_data)
new_data_predictions_svm_libsvm = model_svm_libsvm.predict(new_data)

for i in range(data_sum):
    if new_data_predictions_LogisticRegression[i] == 1:
        print(f"The song: {song_name[i]} of the author: {author_name[i]} in LogisticRegression predicts a new song")
    else:
        print(f"The song: {song_name[i]} of the author: {author_name[i]} in LogisticRegression predicts a old song")

for i in range(data_sum):
    if new_data_predictions_liblinear[i] == 1:
        print(f"The song: {song_name[i]} of the author: {author_name[i]} in svm_Liblinear predicts a new song")
    else:
        print(f"The song: {song_name[i]} of the author: {author_name[i]} in svm_Liblinear predicts a old song")

for i in range(data_sum):
    if new_data_predictions_svm_libsvm[i] == 1:
        print(f"The song: {song_name[i]} of the author: {author_name[i]} in svm_libsvm predicts a new song")
    else:
         print(f"The song: {song_name[i]} of the author: {author_name[i]} in svm_libsvm predicts a old song")


# plotting the the different methods outcomes and compare them
data = {
    'Features type': ['phraseology', 'punctuation', 'lexical usage', 'combined'],
    'LogisticRegression': [0.67,0.72,0.78,0.78],
    'svm_liblinear': [0.56,0.78,0.89,0.78],
    'svm libsvm':[0.94,0.78,0.73,0.94]
}
df = pd.DataFrame(data)
df_melted = pd.melt(df, id_vars='Features type', var_name='Classifier', value_name='Value')

# Create the bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Features type', y='Value', hue='Classifier', data=df_melted)

plt.show()

