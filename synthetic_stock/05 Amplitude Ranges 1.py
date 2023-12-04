import os
import numpy as np
import modules.stock_PCANN as PCANN
from modules.SDGP import load_dataset


# Hyperparameter configuration
config = {}
config["bandwidth"] = 3e-2
config["batch_size"] = 32
config["learning_rate"] = 1e-3
config["epochs"] = 100
config["patience"] = 10

for data_id in range(1, 5):

    # Load the data
    data_dir = "/home/daniel/thesis/data/stock_data/data_{}".format(data_id)
    X, y, L = load_dataset(data_dir)

    # Reconstruction errors
    k = 60 
    E = PCANN.computeReconstructionErrors(k, X)

    # PCANN training
    random_seed = 333
    model, s, history = PCANN.train(E, y, config, random_seed)

    # Save the model
    model_path = f"/home/daniel/thesis/synthetic_stock/trained_models/PCANN_{data_id}"
    model.save(model_path)
    np.save(os.path.join(model_path, "cut_off.npy"), s.numpy())

    # --- PCANN performance
    print("")
    print(f"Anomaly Range {data_id}")
    y_pred_train = PCANN.predict_PCANN(E["train"], model, s)
    y_pred_val = PCANN.predict_PCANN(E["val"], model, s)
    y_pred_test = PCANN.predict_PCANN(E["test"], model, s)
    PCANN.print_performances(y["train"], y_pred_train, set="Training")
    PCANN.print_performances(y["val"], y_pred_val, set="Validation")
    PCANN.print_performances(y["test"], y_pred_test, set="Test")
    print(" ")