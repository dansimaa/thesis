import os
import modules.interest_rates as ir

# Range id
ID = 1

# Anomaly range
anomaly_ranges = {
    "range_0": [0.1, 0.2],
    "range_1": [0.2, 0.3], 
    "range_2": [0.3, 0.4],
    "range_3": [0.4, 0.5]
}

# Settings
seed = 42
ir_dataset = "Euribor_12M.csv"
output_folder = "/home/daniel/thesis/data/IR_data/Euribor12M"
n_anom = 1

# Dataset
range_id = 'range_{}'.format(ID)
rho = anomaly_ranges[range_id]
out_path = os.path.join(output_folder, range_id)
X, y, L = ir.GenerateAnomalousTermStructures(ir_dataset, n_anom, rho, seed, False, out_path)