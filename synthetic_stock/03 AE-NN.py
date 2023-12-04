import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.decomposition import PCA
import modules.stock_PCANN as PCANN
from modules.SDGP import load_dataset

# Load the data
data_dir = "/home/daniel/thesis/data/stock_data/data_0"
X, y, L = load_dataset(data_dir)


# Autoencoder semi-supervised training
X = PCANN.normalized_returns(X)
train_data = tf.cast(X["train"], tf.float32)
test_data = tf.cast(X["test"], tf.float32)
normal_train_data = train_data[y["train"]==0]
autoencoder, cut_off = PCANN.AE(normal_train_data, test_data, 1, 333)
y_train_pred = PCANN.predict_AE(autoencoder, train_data, cut_off)
y_test_pred = PCANN.predict_AE(autoencoder, test_data, cut_off)

print(f'\033[1m Autoencoder \033[0m')
print("----------------------------------------")
PCANN.print_performances(y["train"], y_train_pred, set="Training set")
PCANN.print_performances(y["test"], y_test_pred, set="Test set")


# Reconstruction erros
E = {}
E["train"]= autoencoder(X["train"]).numpy() - X["train"].numpy()
E["val"] = autoencoder(X["val"]).numpy() - X["val"].numpy()
E["test"] = autoencoder(X["test"]).numpy() - X["test"].numpy()

# AENN supervised training
config = {}
config["bandwidth"] = 1e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-3
config["epochs"] = 200
config["patience"] = 10
model, s, history = PCANN.train_AENN(E, y, config, 333)

# AENN performance
y_pred_train = PCANN.predict_PCANN(E["train"], model, s)
y_pred_test = PCANN.predict_PCANN(E["test"], model, s)

print(f'\033[1m AE-NN \033[0m')
print("----------------------------------------")
PCANN.print_performances(y["train"], y_pred_train, set="Training")
PCANN.print_performances(y["val"], PCANN.predict_PCANN(E["val"], model, s), set="Validation")
PCANN.print_performances(y["test"], y_pred_test, set="Test")


# Save the model
autoencoder.save("/home/daniel/thesis/synthetic_stock/trained_models/AENN/AE")
np.save("/home/daniel/thesis/synthetic_stock/trained_models/AENN/AE/cut_off.npy", cut_off)
model.save("/home/daniel/thesis/synthetic_stock/trained_models/AENN/FFNN")
np.save("/home/daniel/thesis/synthetic_stock/trained_models/AENN/cut_off.npy", s.numpy())