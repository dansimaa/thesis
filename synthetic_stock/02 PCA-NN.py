import os
import numpy as np
import modules.stock_PCANN as PCANN
from modules.SDGP import load_dataset

# Output model directory
model_dir = "/home/daniel/thesis/synthetic_stock/trained_models"
model_id = "PCANN_0"

# Load the data
data_dir = "/home/daniel/thesis/data/stock_data/data_0"
X, y, L = load_dataset(data_dir)

# Reconstruction errors
k = 60 
E = PCANN.computeReconstructionErrors(k, X)

# PCANN training
random_seed = 98
config = {}
config["bandwidth"] = 3e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-3
config["epochs"] = 100
config["patience"] = 10
model, s, history = PCANN.train(E, y, config, random_seed)

# PCANN performance
y_pred_train = PCANN.predict_PCANN(E["train"], model, s)
y_pred_test = PCANN.predict_PCANN(E["test"], model, s)
PCANN.print_performances(y["train"], y_pred_train, set="Training")
PCANN.print_performances(y["val"], PCANN.predict_PCANN(E["val"], model, s), set="Validation")
PCANN.print_performances(y["test"], y_pred_test, set="Test")

# Save the model
model_path = os.path.join(model_dir, model_id)
model.save(model_path)
np.save(os.path.join(model_path, "cut_off.npy"), s.numpy())