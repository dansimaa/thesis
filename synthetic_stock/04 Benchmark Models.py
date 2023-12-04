import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM, SVC
import tensorflow as tf

from modules.SDGP import load_dataset
import modules.stock_PCANN as PCANN
from modules.stock_PCANN import print_performances, print_latex


def print_time(ExTime, TestTime):
    print(f'Execution Time: {ExTime}, Test Time: {TestTime}')

# Configuration file
with open("/home/daniel/thesis/config/stock_baseline.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Load the data
data_dir = "/home/daniel/thesis/data/stock_data/data"
X, y, L = load_dataset(data_dir)


#-------------------------
# --- Isolation Forest ---
#-------------------------
t = time.time()
IForest = IsolationForest(random_state=22, **config['IF'])
IForest.fit(X["train"])
y_train_pred = IForest.predict(X["train"])
s = time.time()
y_test_pred = IForest.predict(X["test"])
time_IF, time_test = time.time()-t, time.time()-s

y_train_adj = [1 if x == -1 else 0 for x in y_train_pred]
y_test_adj = [1 if x == -1 else 0 for x in y_test_pred]

print(f'\033[1m Isolation Forest \033[0m')
print("----------------------------------------")
print_time(time_IF, time_test)
print_performances(y["train"], y_train_adj, set="Training set")
print_performances(y["test"], y_test_adj, set="Test set")
print_latex(y["train"], y_train_adj, y["test"], y_test_adj)


# -------------
# ---- LOF ----
# -------------
t = time.time()
LOF = LocalOutlierFactor(**config['LOF'])
LOF.fit(X["train"])
y_train_pred = LOF.predict(X["train"])
s = time.time()
y_test_pred = LOF.predict(X["test"])
time_LOF, time_test = time.time()-t, time.time()-s

y_train_adj = [1 if x == -1 else 0 for x in y_train_pred]
y_test_adj = [1 if x == -1 else 0 for x in y_test_pred]

print(f'\033[1m LOF \033[0m')
print("----------------------------------------")
print_time(time_LOF, time_test)
print_performances(y["train"], y_train_adj, set="Training set")
print_performances(y["test"], y_test_adj, set="Test set")
print_latex(y["train"], y_train_adj, y["test"], y_test_adj)


#---------------------
# ------ DBSCAN ------
#---------------------
t = time.time()
dbscan = DBSCAN(**config['DBSCAN'])
clusters_train = dbscan.fit_predict(X["train"])
s = time.time()
clusters_test = dbscan.fit_predict(X["test"])
time_DBSCAN, time_test = time.time()-t, time.time()-s
y_pred_train = [1 if pred == -1 else 0 for pred in clusters_train]
y_pred_test = [1 if pred == -1 else 0 for pred in clusters_test]

print(f'\033[1m DBSCAN \033[0m')
print("----------------------------------------")
print_time(time_DBSCAN, time_test)
print_performances(y["train"], y_pred_train, set="Training set")
print_performances(y["test"], y_pred_test, set="Test set")
print_latex(y["train"], y_pred_train, y["test"], y_pred_test)


#---------------------
# ------ KMeans ------
#---------------------
# Distance from each point to its nearest cluster center
def find_anomalies(X, kmeans, threshold):
    distances = np.min(kmeans.transform(X), axis=1)
    return distances > threshold

# Determine threshold based on the training set
def determine_threshold(X_train_normal, kmeans, quantile=0.5):
    distances = np.min(kmeans.transform(X_train_normal), axis=1)
    return np.quantile(distances, quantile)

t = time.time()
X_train_normal = X["train"][y["train"] == 0]
kmeans = KMeans(random_state=42, **config["KMeans"]).fit(X_train_normal)
threshold = determine_threshold(X_train_normal, kmeans, 0.5)
s = time.time()

y_test_pred = find_anomalies(X["test"], kmeans, threshold)
time_KMeans, time_test = time.time()-t, time.time()-s
y_train_pred = find_anomalies(X["train"], kmeans, threshold)

print(f'\033[1m KMeans \033[0m')
print("----------------------------------------")
print_time(time_KMeans, time_test)
print_performances(y["train"], y_train_pred, set="Training set")
print_performances(y["test"], y_test_pred, set="Test set")
print_latex(y["train"], y_train_pred, y["test"], y_test_pred)


#---------------------
# ------ OC-SVM ------
#---------------------
t = time.time()
ocsvm = OneClassSVM(**config['OCSVM'])
normal_data = X["train"][y["train"] == 0]
ocsvm.fit(normal_data)

s = time.time()
y_test_pred = ocsvm.predict(X["test"])
time_OCSVM, time_test = time.time()-t, time.time()-s
y_train_pred = ocsvm.predict(X["train"])
y_train_adj = [1 if x == -1 else 0 for x in y_train_pred]
y_test_adj = [1 if x == -1 else 0 for x in y_test_pred]

print(f'\033[1m OC-SVM \033[0m')
print("----------------------------------------")
print_time(time_OCSVM, time_test)
print_performances(y["train"], y_train_adj, set="Training set")
print_performances(y["test"], y_test_adj, set="Test set")
print_latex(y["train"], y_train_adj, y["test"], y_test_adj)


# -------------
# ---- KNN ----
# -------------
t = time.time()
knn = KNeighborsClassifier(**config['KNN'])
knn.fit(X["train"], y["train"])
y_train_pred = knn.predict(X["train"])
s = time.time()
y_test_pred = knn.predict(X["test"])
time_KNN, time_test = time.time()-t, time.time()-s

print(f'\033[1m KNN \033[0m')
print("----------------------------------------")
print_time(time_KNN, time_test)
print_performances(y["train"], y_train_pred, set="Training set")
print_performances(y["test"], y_test_pred, set="Test set")
print_latex(y["train"], y_train_pred, y["test"], y_test_pred)


# -------------
# ---- SVM ----
# -------------
t = time.time()
svm = SVC(**config['SVM'])
svm.fit(X["train"], y["train"])
y_train_pred = svm.predict(X["train"])
s = time.time()
y_test_pred = svm.predict(X["test"])
time_SVM, time_test = time.time()-t, time.time()-s

print(f'\033[1m SVM \033[0m')
print("----------------------------------------")
print_time(time_SVM, time_test)
print_performances(y["train"], y_train_pred, set="Training set")
print_performances(y["test"], y_test_pred, set="Test set")
print_latex(y["train"], y_train_pred, y["test"], y_test_pred)


#--------------------
# --- Autoencoder ---
# -------------------
X = PCANN.normalized_returns(X)
train_data = tf.cast(X["train"], tf.float32)
test_data = tf.cast(X["test"], tf.float32)
normal_train_data = train_data[y["train"]==0]
t = time.time()
autoencoder, cut_off = PCANN.AE(normal_train_data, test_data, 1, 333)
y_train_pred = PCANN.predict_AE(autoencoder, train_data, cut_off)
s = time.time()
y_test_pred = PCANN.predict_AE(autoencoder, test_data, cut_off)
time_AE, time_test = time.time()-t, time.time()-s

print(f'\033[1m Autoencoder \033[0m')
print("----------------------------------------")
print_time(time_AE, time_test)
PCANN.print_performances(y["train"], y_train_pred, set="Training set")
PCANN.print_performances(y["test"], y_test_pred, set="Test set")