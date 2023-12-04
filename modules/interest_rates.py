import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from scipy.integrate import quad
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------
# --- Data Generation Functions ---
# ---------------------------------

# --- Contaminate the term structures
def ContaminateTermStructures(X, n_anom, rho, seed):

    # Set the seed
    np.random.seed(seed)

    # Initialize
    N, M = X.shape                          # N: Number of term structure, M: number of pillars
    AnomalyMask = np.ones((N, M))           # initialize the anomaly mask
    Y = np.zeros((N, M))                    # initialize the labels

    for i in range(N):

        # Sign of the shock ~ U({-1,1})
        u = np.random.rand(n_anom)
        sgn = np.where(u >= 0.5, 1, -1)

        # Amplitude of the shock ~ U([rho_1, rho_2])
        delta = rho[0] + (rho[1] - rho[0])*np.random.rand(n_anom)

        # Location of the shock ~ U({1,...,50}) (choose the pillar)
        J = np.random.randint(0, M, n_anom)

        # Update the anomaly mask and the labels
        for k in range(n_anom):
            AnomalyMask[i, J[k]] = 1 + sgn[k] * delta[k]
            Y[i, J[k]] = 1

    # Compute the contaminated term structure
    Xa = X * AnomalyMask

    return Xa, Y


# --- Concatenate normal and anomalous observations
def concatNormalAnomalous(IR_set, IRa_set, label_set, seed):
    
    # Random seed
    np.random.seed(seed)

    # Number of samples in the set
    n_samples = IR_set.shape[0]

    # Identification labels
    Yu_set = np.zeros(n_samples)
    Yc_set = np.sum(label_set, axis=1)

    # Localization labels
    Lu_set = - np.ones(n_samples)
    Lc_set = np.where(label_set == 1)[1]

    # Concatenate
    IR_conc_set = np.vstack((IR_set, IRa_set))
    Y_conc_set = np.vstack((Yu_set[:, None], Yc_set[:, None]))
    L_conc_set = np.vstack((Lu_set[:, None], Lc_set[:, None]))

    n = IR_conc_set.shape[0]
    shuffled_indices = np.random.permutation(n)
    X_set = IR_conc_set[shuffled_indices]
    y_set = Y_conc_set[shuffled_indices]
    L_set = L_conc_set[shuffled_indices]

    return X_set, y_set.squeeze(), L_set.squeeze()


# --- Save the dataset
def save_dataset(data_dir, X, y, L):
    np.savez(os.path.join(data_dir, 'X.npz'), **X)
    np.savez(os.path.join(data_dir, 'y.npz'), **y)
    np.savez(os.path.join(data_dir, 'L.npz'), **L)


# --- Load the dataset
def load_dataset(data_dir):
    X = np.load(os.path.join(data_dir, 'X.npz'))
    X = {key: X[key] for key in X}
    y = np.load(os.path.join(data_dir, 'y.npz'))
    y = {key: y[key] for key in y}
    L = np.load(os.path.join(data_dir, 'L.npz'))
    L = {key: L[key] for key in L}
    return X, y, L


# --- Data Generation Process 
def GenerateAnomalousTermStructures(ir_dataset, n_anom, rho, seed, save=False, out_path=""):

    # Set the random seed
    np.random.seed(seed)

    # Upload the interest rate dataset
    ir_data_path = "data/IR_data_raw"
    data = pd.read_csv(os.path.join(ir_data_path, ir_dataset))
    data = data.drop('BUSINESS_DATE', axis=1)
    data = data.values

    # 1) Shuffle the data & divide into train, val, test
    n_samples = data.shape[0]
    permuted_indices = np.random.permutation(n_samples)
    split_ratio = [0.7, 0.15, 0.15]
    n_train = int(n_samples * split_ratio[0])
    n_val = int(n_samples * split_ratio[1])
    train_indices = permuted_indices[:n_train]
    val_indices = permuted_indices[n_train:n_train + n_val]
    test_indices = permuted_indices[n_train + n_val:]
    IR_train = data[train_indices]
    IR_val = data[val_indices]
    IR_test = data[test_indices]

    # 2) Contaminate the term structures
    # s1, s2, s3 = 20, 67, 248 (range_1)
    s1, s2, s3 = seed+1, seed+2, seed+3
    IRa_train, Y_train = ContaminateTermStructures(IR_train, n_anom, rho, s1)
    IRa_val, Y_val = ContaminateTermStructures(IR_val, n_anom, rho, s2)
    IRa_test, Y_test = ContaminateTermStructures(IR_test, n_anom, rho, s3)

    # 3) Concatenate normal and contaminated observations
    X, y, L = {}, {}, {}  
    X["train"], y["train"], L["train"] = concatNormalAnomalous(IR_train, IRa_train, Y_train, seed)
    X["val"], y["val"], L["val"] = concatNormalAnomalous(IR_val, IRa_val, Y_val, seed)
    X["test"], y["test"], L["test"] = concatNormalAnomalous(IR_test, IRa_test, Y_test, seed)

    if save:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        save_dataset(out_path, X, y, L)

    return X, y, L



# ---------------------------------
# --- Latent Space Calibration ----
# ---------------------------------

def PCA_calibration(X_train, X_test, k):
    model = PCA(n_components=k)
    model.fit(X_train)
    Xhat_train = model.inverse_transform(model.transform(X_train))
    Xhat_test = model.inverse_transform(model.transform(X_test))
    E_train = Xhat_train - X_train
    E_test = Xhat_test - X_test
    return E_train, E_test


def KDE_rule_of_thumb(X_train, X_test, y_train, y_test, k, flag="OutlierRobust"):

    # --- Compute the reconstruction errors
    E_train, E_test = PCA_calibration(X_train, X_test, k)

    # --- Compute and normalize anomaly scores
    eu_train = np.linalg.norm(E_train[y_train==0,:], axis=1)
    ec_train = np.linalg.norm(E_train[y_train==1,:], axis=1)
    eu_test = np.linalg.norm(E_test[y_test==0,:], axis=1)
    ec_test = np.linalg.norm(E_test[y_test==1,:], axis=1)
    e_max = max(eu_train.max(), ec_train.max())
    e_min = min(eu_train.min(), ec_train.min())
    eu_train = (eu_train - e_min) / (e_max - e_min)
    ec_train = (ec_train - e_min) / (e_max - e_min)
    eu_test = (eu_test - e_min) / (e_max - e_min)
    ec_test = (ec_test - e_min) / (e_max - e_min)

    # Sort the dataset and compute the quartiles
    data = np.sort(np.concatenate((eu_train, ec_train)))
    q3, q1 = np.percentile(data, [75 ,25])
    IQR = q3 - q1
    n = len(data)
    sigma = np.std(data)

    if flag == "Gaussian":
        eta = 1.06 * sigma * n**(-1/5)  
    elif flag == "OutlierRobust":
        eta = 0.79 * IQR * n**(-1/5)
    elif flag == "NonUnimodal":
        eta = 0.9 * min(sigma, IQR/1.34) * n**(-1/5)

    return round(eta, 2)


# Gaussian Kernel
K = lambda u: np.exp(-u**2/2) / np.sqrt(2*np.pi)


# Kernel Density Estimation (vector input version)
def KDE(x, E, eta):
    return np.mean(K( (x[:, None] - E[:, None].T) / eta ), axis=1) / eta


# Kernel Density Estimation (scalar input version)
def KDE_scalar(x, E, eta):
    return np.mean(K((x - E[:,None].T) / eta), axis=1) / eta


def PerformanceCalibration(S, s_hat, y_true):
    y_pred = np.where(S > s_hat, 1, 0)
    performance = {}
    performance["accuracy"] = accuracy_score(y_true, y_pred)
    performance["precision"] = precision_score(y_true, y_pred)
    performance["recall"] = recall_score(y_true, y_pred)
    performance["f1"] = f1_score(y_true, y_pred)
    return performance


def CalibrateLatentSpace(k_values, X_train, X_test, y_train, y_test, eta):

    performance_train = {}
    performance_test = {}

    for k in tqdm(k_values):
        E_train, E_test = PCA_calibration(X_train, X_test, k)
        eu_train = np.linalg.norm(E_train[y_train==0,:], axis=1)
        ec_train = np.linalg.norm(E_train[y_train==1,:], axis=1)
        eu_test = np.linalg.norm(E_test[y_test==0,:], axis=1)
        ec_test = np.linalg.norm(E_test[y_test==1,:], axis=1)
        e_max = max(eu_train.max(), ec_train.max())
        e_min = min(eu_train.min(), ec_train.min())
        eu_train = (eu_train - e_min) / (e_max - e_min)
        ec_train = (ec_train - e_min) / (e_max - e_min)
        eu_test = (eu_test - e_min) / (e_max - e_min)
        ec_test = (ec_test - e_min) / (e_max - e_min)

        def Loss(s):
            AUC_u, _ = quad(lambda w: KDE_scalar(w, eu_train, eta), s, np.inf)
            AUC_c, _ = quad(lambda w: KDE_scalar(w, ec_train, eta), -np.inf, s)
            return AUC_u + AUC_c

        x_0 = 0.5
        res = minimize(Loss, x0=x_0)
        s_hat = res.x[0]

        # --- Print during the calibration
        # print(f'Latent space dimension: {k}')
        # print_results(eu_train, ec_train, eu_test, ec_test, s_hat)

        # Compute and save the performance
        train_set = np.concatenate((eu_train, ec_train))
        train_label = np.concatenate((y_train[y_train==0], y_train[y_train==1]))
        test_set = np.concatenate((eu_test, ec_test))
        test_label = np.concatenate((y_test[y_test==0], y_test[y_test==1]))

        performance_train[k] = PerformanceCalibration(train_set, s_hat, train_label)
        performance_test[k] = PerformanceCalibration(test_set, s_hat, test_label)

    return performance_train, performance_test


def plot_performances(k_values, performance_train, performance_test, target_k=None):
    
    # --- Extract the performances
    arr = np.zeros(len(k_values))
    acc_train, prec_train, rec_train, f1_train = arr.copy(), arr.copy(), arr.copy(), arr.copy()
    acc_test, prec_test, rec_test, f1_test = arr.copy(), arr.copy(), arr.copy(), arr.copy()
    for i, k in enumerate(k_values):
        acc_train[i] = performance_train[k]["accuracy"]
        prec_train[i] = performance_train[k]["precision"]
        rec_train[i] = performance_train[k]["recall"]
        f1_train[i] = performance_train[k]["f1"]
        acc_test[i] = performance_test[k]["accuracy"]
        prec_test[i] = performance_test[k]["precision"]
        rec_test[i] = performance_test[k]["recall"]
        f1_test[i] = performance_test[k]["f1"]

    # --- Plot performances
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(k_values, acc_train, label="Accuracy")
    ax1.scatter(k_values, prec_train, label="Precision")
    ax1.scatter(k_values, rec_train, label="Recall")
    ax1.scatter(k_values, f1_train, label="F1 score")
    if target_k is not None:
        ax1.axvline(x = k_values[k_values==target_k], label=f'k = {target_k}', linestyle="-.")
    ax1.set_ylim([0.1, 1])
    ax1.set_title('Training set')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Evaluation metric')
    ax1.legend(loc='lower right')
    ax2.scatter(k_values, acc_test, label="Accuracy")
    ax2.scatter(k_values, prec_test, label="Precision")
    ax2.scatter(k_values, rec_test, label="Recall")
    ax2.scatter(k_values, f1_test, label="F1 score")
    if target_k is not None:
        ax2.axvline(x = k_values[k_values==target_k], label=f'k = {target_k}', linestyle="-.")
    ax2.set_ylim([0.1, 1])
    ax2.set_title('Test set')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Evaluation metric')
    ax2.legend(loc='lower right')
    plt.subplots_adjust(wspace=0.4)
    plt.show()


def plot_performances_1(k_values, performance_train, performance_test, target_k=None):
    
    # --- Extract the performances
    arr = np.zeros(len(k_values))
    acc_train, prec_train, rec_train, f1_train = arr.copy(), arr.copy(), arr.copy(), arr.copy()
    acc_test, prec_test, rec_test, f1_test = arr.copy(), arr.copy(), arr.copy(), arr.copy()
    for i, k in enumerate(k_values):
        acc_train[i] = performance_train[k]["accuracy"]
        prec_train[i] = performance_train[k]["precision"]
        rec_train[i] = performance_train[k]["recall"]
        f1_train[i] = performance_train[k]["f1"]
        acc_test[i] = performance_test[k]["accuracy"]
        prec_test[i] = performance_test[k]["precision"]
        rec_test[i] = performance_test[k]["recall"]
        f1_test[i] = performance_test[k]["f1"]

    # Set the font properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'cmr10'  
    plt.rcParams['axes.formatter.use_mathtext'] = True

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot training set
    ax1.scatter(k_values, acc_train, label="Accuracy", color='red')
    ax1.scatter(k_values, prec_train, label="Precision", color='blue')
    ax1.scatter(k_values, rec_train, label="Recall", color='orange')
    ax1.scatter(k_values, f1_train, label="F1 score", color='black')
    if target_k is not None:
        ax1.axvline(x = k_values[k_values==target_k], label=f'k = {target_k}', linestyle="-.")
    ax1.set_ylim([0.1, 1])
    ax1.set_title('Training set', fontsize=22)
    ax1.set_xlabel('Latent Space Dimension k', fontsize=18)
    ax1.set_ylabel('Evaluation Metric', fontsize=18)
    ax1.legend(loc='lower center', fontsize=18)

    # Plot test set
    ax2.scatter(k_values, acc_test, label="Accuracy", color='red')
    ax2.scatter(k_values, prec_test, label="Precision", color='blue')
    ax2.scatter(k_values, rec_test, label="Recall", color='orange')
    ax2.scatter(k_values, f1_test, label="F1 score", color='black')
    if target_k is not None:
        ax2.axvline(x = k_values[k_values==target_k], label=f'k = {target_k}', linestyle="-.")
    ax2.set_ylim([0.1, 1])
    ax2.set_title('Test set', fontsize=22)
    ax2.set_xlabel('Latent Space Dimension k', fontsize=18)
    ax2.set_ylabel('Evaluation Metric', fontsize=18)
    ax2.legend(loc='lower center', fontsize=18)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2)

    # Show the plot
    plt.show()



# --- Upload the dataset
def upload_IR_dataset(data_path):
    X, y, L = {}, {}, {}
    X["train"] = np.genfromtxt(os.path.join(data_path, 'X_train.csv'), delimiter=',')
    y["train"] = np.genfromtxt(os.path.join(data_path, 'y_train.csv'), delimiter=',')
    X["val"] = np.genfromtxt(os.path.join(data_path, 'X_val.csv'), delimiter=',')
    y["val"] = np.genfromtxt(os.path.join(data_path, 'y_val.csv'), delimiter=',')
    X["test"]  = np.genfromtxt(os.path.join(data_path, 'X_test.csv'), delimiter=',')
    y["test"] = np.genfromtxt(os.path.join(data_path, 'y_test.csv'), delimiter=',')
    return X, y, L