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
from stylometry.classify import *
from stylometry.cluster import *


novel_corpus = StyloCorpus.from_glob_pattern('stylometry-data/*/*.txt')
novel_corpus.output_csv('novels.csv')

# Create a KMeans clusterer and run PCA on the data
# Load data from CSV file old.csv
data = pd.read_csv('novels.csv',encoding="windows-1255")
# removing last col of the file - contains garbege cause of the hebrew

data = data.iloc[:, :-1]
for i in range(20):
    data['Author'][i] = 'new'

for i in range(38):
    data['Author'][i+20] = 'old'

data.drop(columns=['Title'], inplace=True)

data.to_csv('novels.csv', index=False,encoding="utf-8")

kmeans = StyloKMeans('novels.csv')
# Cluster the PCA'd data using K-means
kmeans.fit()
# Shot the plot of explained variance per principle component
print(kmeans.stylo_pca.plot_explained_variance())
# Show the plot of the PCA'd data with the cluster centroids
kmeans.plot_clusters()

