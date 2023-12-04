import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import modules.interest_rates as ir
import modules.interest_rates_models as irm

ID = 0
seed = 333

# Upload the dataset
data_path = "data/IR_data/Euribor12M/range_{}".format(ID)
X, y, L = ir.load_dataset(data_path)

# Autoencoder semi-supervised training
train_data = tf.cast(X["train"], tf.float32)
test_data = tf.cast(X["test"], tf.float32)
normal_train_data = train_data[y["train"]==0]
t_1 = time.time()
autoencoder, _ = irm.AE(normal_train_data, test_data, 1, seed)
t_AE = time.time() - t_1

# Reconstruction erros
E = {}
E["train"]= autoencoder(X["train"]).numpy() - X["train"]
E["val"] = autoencoder(X["val"]).numpy() - X["val"]
E["test"] = autoencoder(X["test"]).numpy() - X["test"]

# FFNN supervised training
config = {}
config["bandwidth"] = 1e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-4
config["epochs"] = 1000
config["patience"] = 20
t_2 = time.time() 
model, s, history = irm.train(E, y, config, seed)

# Performance
y_pred_train = irm.predict_PCANN(E["train"], model, s)

t_3 = time.time()
y_pred_test = irm.predict_PCANN(E["test"], model, s)
train_time, test_time = t_AE + time.time() - t_2, time.time() - t_3


y_pred_val = irm.predict_PCANN(E["val"], model, s)
print(f"Train time: {train_time}, Test time: {test_time}")
irm.print_performances(y["train"], y_pred_train, set="Training")
irm.print_performances(y["val"], y_pred_val, set="Validation")
irm.print_performances(y["test"], y_pred_test, set="Test")

# Save the models
save_path = "/home/daniel/thesis/interest_rates/trained_models/AENN_{}".format(ID)
autoencoder.save(os.path.join(save_path, "AE"))
model.save(os.path.join(save_path, "FFNN"))
np.save(os.path.join(save_path, "cut_off.npy"), s.numpy())