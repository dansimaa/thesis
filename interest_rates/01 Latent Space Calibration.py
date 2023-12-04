import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import modules.interest_rates as ir


# Range id
ID = 3

# Upload the dataset
data_path = "data/IR_data/Euribor12M/range_{}".format(ID)
X, y, L = ir.load_dataset(data_path)
X["test"] = np.vstack((X["test"], X["val"]))
y["test"] = np.ravel(np.vstack((y["test"][:, None], y["val"][:, None])))

# Bandwith parameter 
eta = ir.KDE_rule_of_thumb(X["train"], X["test"], y["train"], y["test"], 10)
print(f"Bandiwidth parameter: {eta}")

# Latent space range performances
k_values = np.arange(2, 52, 2) 
performance_train, performance_test = ir.CalibrateLatentSpace(k_values, 
                                                              X["train"], X["test"],
                                                              y["train"], y["test"], 
                                                              eta)

# Plot performances
target_k = 12
ir.plot_performances_1(k_values, performance_train, performance_test, target_k)