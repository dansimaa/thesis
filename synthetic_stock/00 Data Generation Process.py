from modules.SDGP import data_generation_process, save_dataset

# Anomaly range
range_id = 0

# Data directory
data_dir = "/home/daniel/thesis/data/stock_data/data_{}".format(range_id)

# Anamaly amplitude ranges
amplitude_ranges = {
    0: (0.000, 0.100),
    1: (0.000, 0.025),
    2: (0.025, 0.050),
    3: (0.050, 0.075),
    4: (0.075, 0.100)
}

# Data generation
random_seed = 42
rho_1, rho_2 = amplitude_ranges[range_id]
X, y, L = data_generation_process(random_seed, rho_1, rho_2)
save_dataset(data_dir, X, y, L)