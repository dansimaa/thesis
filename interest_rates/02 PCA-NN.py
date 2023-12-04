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


ID = 1
SEED = 333

# Upload the dataset
data_path = "data/IR_data/Euribor12M/range_{}".format(ID)
X, y, L = ir.load_dataset(data_path)

# Reconstruction errors
E = {}
k = 12
E["train"], E["val"], E["test"] = irm.computeReconstructionErrors(k, X)

# PCANN training
config = {}
config["bandwidth"] = 1e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-4
config["epochs"] = 1000
config["patience"] = 20

t = time.time()
model, s, history = irm.train(E, y, config, SEED)

# Performance
y_pred_train = irm.predict_PCANN(E["train"], model, s)
st = time.time()
y_pred_test = irm.predict_PCANN(E["test"], model, s)
train_time, test_time = time.time()-t, time.time()-st
y_pred_val = irm.predict_PCANN(E["val"], model, s)

print(f"Train time: {train_time}, Test time: {test_time}")
irm.print_performances(y["train"], y_pred_train, set="Training")
irm.print_performances(y["val"], y_pred_val, set="Validation")
irm.print_performances(y["test"], y_pred_test, set="Test")

# Save the model
model_path = "/home/daniel/thesis/interest_rates/trained_models/PCANN_{}".format(ID)
model.save(model_path)
np.save(os.path.join(model_path, "cut_off.npy"), s.numpy())