import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM, SVC
import tensorflow as tf

import modules.interest_rates as ir
from modules.interest_rates_models import *


ID = 1

# Configuration file
with open("/home/daniel/thesis/config/ir_baseline.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Upload the dataset
data_path = "data/IR_data/Euribor12M/range_{}".format(ID)
X, y, L = ir.load_dataset(data_path)


def print_time(ExTime, TestTime):
    print(f'Execution Time: {ExTime}, Test Time: {TestTime}')


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
threshold = determine_threshold(X_train_normal, kmeans, 0.1)
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


#--------------------
# --- Autoencoder ---
# -------------------
train_data = tf.cast(X["train"], tf.float32)
test_data = tf.cast(X["test"], tf.float32)
train_labels = y["train"].astype(bool)
test_labels = y["test"].astype(bool)
normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]
anomalous_train_data = train_data[train_labels]
anomalous_test_data = test_data[test_labels]

t = time.time()
autoencoder, cut_off = AE(normal_train_data, test_data, 3.0, 333)
y_train_pred = predict_AE(autoencoder, train_data, cut_off)
s = time.time()
y_test_pred = predict_AE(autoencoder, test_data, cut_off)
time_AE, time_test = time.time()-t, time.time()-s

print(f'\033[1m Autoencoder \033[0m')
print("----------------------------------------")
print_time(time_AE, time_test)
print_performances(train_labels, y_train_pred, set="Training set")
print_performances(test_labels, y_test_pred, set="Test set")
print_latex(train_labels, y_train_pred, test_labels, y_test_pred)