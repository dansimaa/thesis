# Synthetic Data Generation Process
# Generate a synthetic dataset, that is realistic, to train an anomaly detection model. 
# This is done since anomalies are rare events by definition, this means that real datasets
# are very imbalanced and the training phase becames more difficult. The following synthetic
# data generation is a three step process: data simulation, data contamination and data augmentation. 


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import json


# Black & Scholes correlated Monte Carlo
def BS_MC(S0, mu, sigma, corr, T, M, seed=42):
    """
    Monte Carlo simulations of the stocks' prices of correlated assets under the
    Black and Scholes model.

    Parameters
    - S0:     initial stocks' prices
    - mu:     stocks' drifts
    - sigma:  stocks' volatilities
    - corr:   correlation matrix
    - T:      time period (number of years)
    - M:      number of time steps
    """
    np.random.seed(seed)
    N = len(S0)
    dt = T / M
    C = np.linalg.cholesky(corr)
    X = np.zeros((N, M+1))

    for i in range(M):
        Z = np.random.randn(N)
        Y = np.matmul(C, Z)
        X[:, i+1] = X[:, i] + (mu - sigma**2/2)*dt + sigma*np.sqrt(dt)*Y

    S = S0[:, None] * np.exp(X[:, 1:])

    return S




# Introduce anomalies in the time series (Algorithm 6)
def DataContamination(S, n, rho, seed=42):
    """
    Given a set of time series S, introduce a fixed number of anomalies
    n, by applying a shock on some original values of the observed time
    series.

    Parameters
    - S:    data (each row corresponds to a time series)
    - n:    number of anomalies to introduce in each time series
    - rho:  upperbound of the shock amplitude

    Output
    - Sa:   time series with anomalies
    - Y:    labels associated with each value of the time series
    """

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Initialize
    N, T = S.shape                  # N: number of time series, T: number of time steps
    AnomalyMask = np.ones((N, T))   # initialize the anomaly mask
    Y = np.zeros((N, T))            # initialize the labels

    # Algorithm 6 (p. 12)
    # Build the anomaly mask and the labels, loop for each stock
    for i in range(N):
        # Sign of the shock, ~ U({-1,1})
        u = np.random.rand(n)
        sgn = np.where(u >= 0.5, 1, -1)

        # Amplitude of the shock, ~ U([0,rho])
        delta = rho * np.random.rand(n)

        # Location of the shock, ~U({1,...,T})
        J = np.random.randint(0, T, n)

        # Update the anomaly mask and the labels
        for k in range(n):
            AnomalyMask[i, J[k]] = 1 + sgn[k] * delta[k]
            Y[i, J[k]] = 1

    # Compute the contaminated time series
    Sa = S * AnomalyMask

    return Sa, Y




# Introduce anomalies in the time series (Algorithm 6)
def BoundedDataContamination(S, n, rho_1, rho_2, seed=42):
    """
    Given a set of time series S, introduce a fixed number of anomalies
    n, by applying a shock on some original values of the observed time
    series.

    Parameters
    - S:    data (each row corresponds to a time series)
    - n:    number of anomalies to introduce in each time series
    - rho:  upperbound of the shock amplitude

    Output
    - Sa:   time series with anomalies
    - Y:    labels associated with each value of the time series
    """

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Initialize
    N, T = S.shape                  # N: number of time series, T: number of time steps
    AnomalyMask = np.ones((N, T))   # initialize the anomaly mask
    Y = np.zeros((N, T))            # initialize the labels

    # Algorithm 6 (p. 12)
    # Build the anomaly mask and the labels, loop for each stock
    for i in range(N):
        # Sign of the shock, ~ U({-1,1})
        u = np.random.rand(n)
        sgn = np.where(u >= 0.5, 1, -1)

        # Amplitude of the shock, ~ U([0,rho])
        delta = rho_1 + (rho_2 - rho_1) * np.random.rand(n)

        # Location of the shock, ~U({1,...,T})
        J = np.random.randint(0, T, n)

        # Update the anomaly mask and the labels
        for k in range(n):
            AnomalyMask[i, J[k]] = 1 + sgn[k] * delta[k]
            Y[i, J[k]] = 1

    # Compute the contaminated time series
    Sa = S * AnomalyMask

    return Sa, Y


# Data Augmentation with sliding window technique (Algorithm 7)
def DataAugmentation(S, Y, p):
    """
    Apply the sliding window data augmentation technique: given a time
    series of length T, extract T-p+1 sub-time series of length p.
    It allows to increase considerably the number of anomalies and reduce
    the dependency of the event location.

    Parameters
    - S:       data (N,T), (each row corresponds to a time series)
    - Y:       labels (N,T), associated with each value of the time series
    - p:       window size

    Returns
    - Xs:      slided time series (N*(T-p+1), T)
    - Ys:      slided labels (N*(T-p+1), T)
    """

    # Initialize
    N, T = S.shape
    Np = T - p + 1
    Xs = np.zeros((Np * N, p))
    Ys = np.zeros((Np * N, p))

    # Data Augmentation with sliding window technique
    k = 0
    for i in range(N):
        for q in range(Np):
            Xs[k,:] = S[i, q:(q+p)]
            Ys[k,:] = Y[i, q:(q+p)]
            k += 1

    return Xs, Ys




# Time series selection and labeling function (Algorithm 8 and Algorithm 9)
def SelectionAndLabelling(Xs, Ys, flag, rc=0):
    """
    Given a dataset of time series with anomalies (contaminated) and without (uncontaminated),
    return a balanced dataset by randomly discarding the uncontaminated ones, until the number
    of instances in the two classes are equal. Moreover by assumption, keep only the time series
    with at most one anomaly.

    Parameters
    - Xs:         slided time series data
    - Ys:         slided labels
    - flag:       0 training set, 1 test set
    - rc:         contamination rate (for test set)

    Returns
    - X:          time series with at most one anomaly
    - A:          labels for identification task
    - L:          labels for localization task
    """

    # Algorithm 8
    # Keep time series with at most one anomaly
    index = np.where(np.sum(Ys, axis=1) <= 1)[0]
    Xs = Xs[index, :]
    Ys = Ys[index, :]

    # Training set
    if flag == 0:
        Nc = np.sum(np.sum(Ys, axis=1) == 1)                      # number of contaminated time series
        Nu = np.sum(np.sum(Ys, axis=1) == 0)                      # number of uncontaminated time series
        indeces = np.where(np.sum(Ys, axis=1) == 0)[0]            # indexes of the uncontaminated time series
        index_u = indeces[np.random.permutation(Nu)[:Nc]]         # randomly select Nc from the Nu indexes of the uncontaminated time series

    # Test set
    elif flag == 1:
        Nc = np.sum(np.sum(Ys, axis=1) == 1)                      # number of contaminated time series
        Nu = np.sum(np.sum(Ys, axis=1) == 0)                      # number of uncontaminated time series
        Nu_hat = int(np.ceil(Nc * (1-rc)/rc))                     # choosed number of uncontaminated time series
        indeces = np.where(np.sum(Ys, axis=1) == 0)[0]            # indexes of the uncontaminated time series
        index_u = indeces[np.random.permutation(Nu)[:Nu_hat]]

    index_c = np.where(np.sum(Ys, axis=1) == 1)[0]                # indexes of the uncontaminated time series
    index = np.union1d(index_u, index_c)                          # indexes to generate the balanced dataset
    X = Xs[index, :]
    Ys = Ys[index, :]


    # Algorithm 9: Time series labelling
    # Identification label
    A = np.sum(Ys, axis=1)

    # Localization label
    L = - np.ones(A.shape)                                        # initialize the label with -1
    idx = np.where(A==1)[0]                                       # indexes that I have to update with the location
    BooleanMatrix = ( Ys[A == 1, :] == np.max(Ys[A == 1, :], axis=1)[:, None] )
    for i in range(len(idx)):
        L[idx[i]] = int( np.where(BooleanMatrix[i, :])[0][0] )

    return X, A, L




#################################################################
##################### Utility functions #########################
#################################################################

# Plot the sample path simulations
def plot_paths(S):

    # Create a new figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the color cycle for the plot
    ax.set_prop_cycle(color=plt.cm.Dark2(np.linspace(0, 1, S.shape[0])))

    # Iterate over each row of the matrix S and plot the time series
    for i in range(len(S)):
        ax.plot(S[i], linestyle='-', marker='')

    ax.set_title("Stock's path simulations")
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock price')
    plt.show()


# Plot contaminated time series
def plot_contaminated_paths(Sa):
    plt.figure(figsize=(12,6))
    plt.plot(Sa.T)   # Sa.T[:,i-th tm] see the i-th
    plt.title('Contaminated Stock price')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


# Plot each contaminated time series
def plot_every_contaminated_paths(Sa, Y, T, M):
    
    # Time discretization parameters
    t = np.linspace(0, T, M)

    # Plot the contaminated time series
    for J in range(Sa.shape[0]):
        plt.figure(figsize=(12,8))
        index = np.where(Y[J,:]==1)
        plt.plot(t, Sa.T[:,J])   # Sa.T[:,i-th tm] see the i-th
        plt.scatter(t[index], Sa.T[index, J].flatten(), marker='x', s=100, color='orange')
        plt.title(f'Contaminated Stock price {J+1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()


# Check that the labels are coherent
def check_set(X, A, L, set='Training'):
    print(set)
    print('------------------------------')
    print(f'Number of time series: {X.shape[0]}')
    print(f'Contaminated: {np.sum(A==1)}')
    print(f'Uncontaminated: {np.sum(A==0)}')
    print(f'Number of localization labels: {np.sum(L != -1)}')
    check = np.array_equal(np.where(A==0)[0], np.where(L==-1)[0])
    print(f'Check on the indexes: {check}')
    print(' ')


# Shuffle the datasets
def shuffle_data(X, A, L, seed):    
    
    np.random.seed(seed)                              # Set the seed
    length = X.shape[0]                             # Number of samples
    idx_shuffled = np.random.permutation(length)    # Randomly shuffle the indexes

    # Shuffle the data
    X = X[idx_shuffled, :]  # shuffle the time series
    A = A[idx_shuffled]     # shuffle the identification labels
    L = L[idx_shuffled]     # shuffle the localization label

    return X, A, L



# --- Generate a dataset
def data_generation_process(seed, rho_1, rho_2):
    np.random.seed(42)

    # Data Simulation
    N = 20                                          
    T = 6                                           
    M = 1500                                        
    S0 = 100 + np.random.randn(N)                   
    mu = 0.01 + (0.2-0.01)*np.random.rand(N)        
    sigma = 0.01 + (0.1-0.01)*np.random.rand(N)   
    corr = np.genfromtxt('/home/daniel/thesis/data/stock_data/MyCorrMatrix', delimiter=',')  
    S = BS_MC(S0, mu, sigma, corr, T, M, seed)
    # plot_paths(S)

    # Data Contamination
    rc = 0.5                             
    n_anom = 4                             
    Sa, Y = BoundedDataContamination(S, n_anom, rho_1, rho_2, seed)   
    # plot_contaminated_paths(Sa)

    # Data Augmentation and Selection
    p = 206
    S_train, Y_train  = Sa[:, :900], Y[:, :900]
    S_val, Y_val = Sa[:, 900:1200], Y[:, 900:1200]
    S_test, Y_test = Sa[:, 1200:], Y[:, 1200:]

    Xs_train, Ys_train = DataAugmentation(S_train, Y_train, p)
    Xs_val, Ys_val = DataAugmentation(S_val, Y_val, p)
    Xs_test, Ys_test = DataAugmentation(S_test, Y_test, p)

    X_train, A_train, L_train = SelectionAndLabelling(Xs_train, Ys_train, 0, rc)
    X_val, A_val, L_val = SelectionAndLabelling(Xs_val, Ys_val, 0, rc)
    X_test, A_test, L_test = SelectionAndLabelling(Xs_test, Ys_test, 1, rc)

    # Shuffle and save the data
    X, y, L = {}, {}, {}
    X["train"], y["train"], L["train"] = shuffle_data(X_train, A_train, L_train, seed)
    X["val"], y["val"], L["val"] = shuffle_data(X_val, A_val, L_val, seed)
    X["test"], y["test"], L["test"] = shuffle_data(X_test, A_test, L_test, seed)

    return X, y, L


def save_dataset(data_dir, X, y, L):
    np.savez(os.path.join(data_dir, 'X.npz'), **X)
    np.savez(os.path.join(data_dir, 'y.npz'), **y)
    np.savez(os.path.join(data_dir, 'L.npz'), **L)


def load_dataset(data_dir):
    X = np.load(os.path.join(data_dir, 'X.npz'))
    X = {key: X[key] for key in X}
    y = np.load(os.path.join(data_dir, 'y.npz'))
    y = {key: y[key] for key in y}
    L = np.load(os.path.join(data_dir, 'L.npz'))
    L = {key: L[key] for key in L}
    return X, y, L