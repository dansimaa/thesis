import os
import numpy as np
import json
import modules.stock_PCANN as PCANN
from modules.SDGP import data_generation_process


# Output 
performances_1, performances_2 = {}, {}
output_path = "/home/daniel/thesis/synthetic_stock/model_performances/amplitude_ranges"
path_1 = os.path.join(output_path, "identification_p1.json")
path_2 = os.path.join(output_path, "localization_p1.json")

# Anomaly amplitude ranges
rho_values = [0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]   

# Hyperparameter configuration
config = {}
config["bandwidth"] = 3e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-3
config["epochs"] = 100
config["patience"] = 10

# Training
for i, rho in enumerate(rho_values):

    # Data generation
    random_seed = 42 + 10*i
    X, y, L = data_generation_process(random_seed, 0.0, rho) 

    # Reconstruction errors
    E = PCANN.computeReconstructionErrors(60, X)

    # PCANN training
    model, s, _ = PCANN.train(E, y, config, 333)

    # Identification performances
    perf_1 = {}
    perf_1["train"] = PCANN.EvaluatePerformance(y["train"], PCANN.predict_PCANN(E["train"], model, s))
    perf_1["val"] = PCANN.EvaluatePerformance(y["val"], PCANN.predict_PCANN(E["val"], model, s))
    perf_1["test"] = PCANN.EvaluatePerformance(y["test"], PCANN.predict_PCANN(E["test"], model, s))
    performances_1[rho] = perf_1
            
    with open(path_1, 'w') as f:
        json.dump(performances_1, f)

    # Localization performances
    perf_2 = {}
    perf_2["train"] = PCANN.localization(X["train"], y["train"], L["train"], E["train"])
    perf_2["val"] = PCANN.localization(X["val"], y["val"], L["val"], E["val"])
    perf_2["test"] = PCANN.localization(X["test"], y["test"], L["test"], E["test"])
    performances_2[rho] = perf_2

    with open(path_2, 'w') as f:
        json.dump(performances_2, f)