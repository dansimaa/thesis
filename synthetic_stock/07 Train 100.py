import os
import json
import modules.stock_PCANN as PCANN
from modules.SDGP import data_generation_process

# Output paths
output_path = "/content/drive/MyDrive/THESIS/thesis/synthetic_stock/model_performances/test_100"
perf_1, perf_2, perf_3 = [], [], []
path_1 = os.path.join(output_path, "identification.json")
path_2 = os.path.join(output_path, "localization_1.json")
path_3 = os.path.join(output_path, "localization_2.json")

# PCANN hyperparameters
config = {}
config["bandwidth"] = 3e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-3
config["epochs"] = 100
config["patience"] = 10

for i in range(100):
    ide, loc1, loc2 = {}, {}, {}

    random_seed = 132 + 11*i  
    X, y, L = data_generation_process(random_seed, 0.0, 0.1)
    E = PCANN.computeReconstructionErrors(60, X)
    model, s, _ = PCANN.train(E, y, config, random_seed)  

    ide["train"] = PCANN.EvaluatePerformance(y["train"], PCANN.predict_PCANN(E["train"], model, s))
    ide["val"] = PCANN.EvaluatePerformance(y["val"], PCANN.predict_PCANN(E["val"], model, s))
    ide["test"] = PCANN.EvaluatePerformance(y["test"], PCANN.predict_PCANN(E["test"], model, s))
    perf_1.append(ide)      
    with open(path_1, 'w') as f:
        json.dump(perf_1, f)

    loc1["train"] = PCANN.localization(X["train"], y["train"], L["train"], E["train"])
    loc1["val"] = PCANN.localization(X["val"], y["val"], L["val"], E["val"])
    loc1["test"] = PCANN.localization(X["test"], y["test"], L["test"], E["test"])
    perf_2.append(loc1) 
    with open(path_2, 'w') as f:
        json.dump(perf_2, f)

    loc2["train"] = PCANN.localization(X["train"], y["train"], L["train"], E["train"], True)
    loc2["val"] = PCANN.localization(X["val"], y["val"], L["val"], E["val"], True)
    loc2["test"] = PCANN.localization(X["test"], y["test"], L["test"], E["test"], True)
    perf_3.append(loc2) 
    with open(path_3, 'w') as f:
        json.dump(perf_3, f)