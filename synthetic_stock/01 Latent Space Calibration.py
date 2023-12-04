import numpy as np
from modules.SDGP import load_dataset
from modules.interest_rates import KDE_rule_of_thumb, CalibrateLatentSpace, plot_performances

# Load the data
data_dir = "/home/daniel/thesis/data/stock_data/data_0"
X, y, L = load_dataset(data_dir)
X["test"] = np.vstack((X["test"], X["val"]))
y["test"] = np.ravel(np.vstack((y["test"][:, None], y["val"][:, None])))

# Bandwith parameter 
eta = KDE_rule_of_thumb(X["train"], X["test"], y["train"], y["test"], 10)
print(f"Bandiwidth parameter: {eta}")

#  Latent space range performances
k_values = np.arange(5, 205, 5)
performance_train, performance_test = CalibrateLatentSpace(k_values,
                                                           X["train"], X["test"],
                                                           y["train"], y["test"],
                                                           eta)

# Plot performances
target_k = 60
plot_performances(k_values, performance_train, performance_test, target_k)