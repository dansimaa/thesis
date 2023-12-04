import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- PCA-based reconstruction errors 
def computeReconstructionErrors(k, X):
    X_train, X_val, X_test = X["train"], X["val"], X["test"]
    model = PCA(n_components=k, random_state=42)
    model.fit(X_train)
    Xhat_train = model.inverse_transform(model.transform(X_train))
    Xhat_val = model.inverse_transform(model.transform(X_val))
    Xhat_test = model.inverse_transform(model.transform(X_test))
    E_train = Xhat_train - X_train
    E_val = Xhat_val - X_val
    E_test = Xhat_test - X_test
    return E_train, E_val, E_test


# --- Kernel Density Estimation
def KDE(x, E, eta):
    GaussianKernel = lambda t: tf.math.exp(-t**2/2)/tf.math.sqrt(2*tf.constant(np.pi))
    return tf.reduce_mean( GaussianKernel( ( tf.reshape(x,[-1,1]) - tf.reshape(E,[1,-1]) ) / eta ), axis=1 ) / eta


# --- Numerical integration
def AUCDensity(s, scores_u, scores_c, eta, n=100):
    xu = tf.linspace(s, 2.0, n)
    step_u = xu[1] - xu[0]
    fu = KDE(xu, scores_u, eta)
    AUC_u = 0.5 * step_u * tf.math.reduce_sum( fu[:-1] + fu[1:])
    xc = tf.linspace(-1.0, s, n)
    step_c = xc[1] - xc[0]
    fc = KDE(xc, scores_c, eta)
    AUC_c = 0.5 * step_c * tf.math.reduce_sum( fc[:-1] + fc[1:])
    return AUC_u, AUC_c


# --- Extract a random batch
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def print_status_bar(iteration, total, loss, metrics=None, val_loss=None, val_metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    if val_loss is not None and val_metrics is not None:
        val_metrics = " - ".join(["val_loss: " + str(val_loss.numpy())]+["val_{}: {:.4f}".format(m.name, m.result()) for m in val_metrics])
        metrics += " - " + val_metrics
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)


#------------Print the results
def plot_KDE(scores_u, scores_c, s, eta):

    # Convert tensor to numpy
    scores_u = scores_u.numpy()
    scores_c = scores_c.numpy()
    s = s.numpy()

    # Gaussian kernel, KDE uncontaminated and contaminated
    K = lambda t: np.exp(-t**2/2)/np.sqrt(2*np.pi)
    def KernelDensityU(x):
        return np.mean( K(( x[:,None] - scores_u.T )/eta), axis=1 ) / eta
    def KernelDensityC(x):
        return np.mean( K(( x[:,None] - scores_c.T )/eta), axis=1 ) / eta

    #plt.figure(figsize=(8,5))
    plt.hist(scores_u, density=True, bins=20, alpha=0.5, label='Uncontaminated', color='black')
    plt.hist(scores_c, density=True, bins=20, alpha=0.5, label='Contaminated', color='red')

    E = np.arange(min(scores_u.min(), scores_c.min()), max(scores_u.max(), scores_c.max()), 0.01)
    plt.plot(E, KernelDensityU(E), color='black')
    plt.plot(E, KernelDensityC(E), color='red')
    plt.axvline(x = s, color = 'orange', label='Cut-off')
    plt.xlabel('Reconstruction errors')
    plt.ylabel('Density')
    plt.legend(loc='upper right')

    # Show the plot
    plt.show()





def train(E, y, config, seed): 

    # Random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Hyperparameters
    eta = config["bandwidth"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    n_epochs = config["epochs"]
    patience = config["patience"]
    n_steps = len(E["train"]) // batch_size

    # Shallow neural network model
    model = keras.Sequential([
        keras.layers.Input(shape=(50,)),
        keras.layers.Dense(25, activation='relu'), 
        keras.layers.Dense(12, activation='relu'), 
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Optimizer, Loss, Metrics
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.binary_crossentropy
    mean_loss = keras.metrics.Mean(name="loss")
    metrics = [keras.metrics.BinaryAccuracy()]   # keras.metrics.Recall()
    val_metrics = [keras.metrics.BinaryAccuracy()]

    # Training history
    history = {'loss': [], 'val_loss': []}
    for metric in metrics:
        history[metric.name] = []
    for metric in val_metrics:
        history['val_' + metric.name] = []

    # Early stopping
    best_val_loss = float('inf')
    best_weights = None
    best_s = tf.Variable(0.0)   
    wait = 0

    # Cut-off value
    s = tf.Variable(0.5) 
    eps = 1e-7

    # Customized training loop
    for epoch in range(1, n_epochs + 1):                                            
        print("Epoch {}/{}".format(epoch, n_epochs))                                
        for step in range(1, n_steps + 1):                                          
            X_batch, y_batch = random_batch(E["train"], y["train"][:,None], batch_size)   

            with tf.GradientTape(persistent=True) as tape:
                scores = model(X_batch, training=True)
                y_pred = tf.where( scores>s , 1-eps, eps)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                AUC_u, AUC_c = AUCDensity(s, scores[y_batch==0], scores[y_batch==1], eta)
                loss_integral = AUC_u + AUC_c
                loss = tf.add_n([main_loss] + model.losses + [loss_integral])

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            gradient_s = tape.gradient(loss_integral, s)
            optimizer.apply_gradients([(gradient_s, s)])
            del tape

            mean_loss(loss)
            for metric in metrics:
                metric(y_batch, y_pred)
            print_status_bar(step * batch_size, len(y["train"]), mean_loss, metrics)

        if s.numpy() < 0:
            s.assign(0.5)

        # --- Validation step
        scores_val = model(E["val"], training=False)
        y_val_pred = tf.where( scores_val>s , 1-eps, eps)
        main_loss = tf.reduce_mean(loss_fn(y["val"][:, None], y_val_pred))
        AUC_u, AUC_c = AUCDensity(s, scores_val[y["val"]==0], scores_val[y["val"]==1], eta)
        val_loss = main_loss + AUC_u + AUC_c
        for metric in val_metrics:
            metric(y["val"], y_val_pred)

        # --- Save the history
        history['loss'].append(mean_loss.result().numpy())
        history['val_loss'].append(val_loss.numpy())
        for metric in metrics:
            history[metric.name].append(metric.result().numpy())
        for metric in val_metrics:
            history['val_' + metric.name].append(metric.result().numpy())

        # ------------------------------------------- End epoch prints
        print_status_bar(len(y["train"]), len(y["train"]), mean_loss, metrics, val_loss, val_metrics)
        # print(f'\ns: {s.numpy(): .4f}, AUC_u: {AUC_u.numpy(): .4f}, AUC_c: {AUC_c.numpy(): .4f}, total: {AUC_u.numpy()+AUC_c.numpy():.4f}')
        # plot_KDE(model(E_train)[y_train==0], model(E_train)[y_train==1], s)

        # --- Reset the metrics at the end of the epoch
        for metric in [mean_loss] + metrics + val_metrics:
            metric.reset_states()

        # --- Early stopping
        wait += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            wait = 0
            best_s.assign( s.numpy() )
        if wait >= patience:
            break

    # --- Save best weights and cut-off value
    model.set_weights(best_weights)     
    s.assign( best_s.numpy() ) 

    return model, s, history  



# --- PCANN prediction 
def predict_PCANN(X_set, model, s):
    scores = model.predict(X_set, verbose=0)
    y_pred = np.round(scores > s.numpy()).flatten()
    return y_pred


# --- PCANN evaluation
def EvaluatePerformance(y_true, y_pred):
    perf = {}
    perf["accuracy"] = accuracy_score(y_true, y_pred)
    perf["precision"] = precision_score(y_true, y_pred)
    perf["recall"] = recall_score(y_true, y_pred)
    perf["f1"] = f1_score(y_true, y_pred)
    return perf


def PrintPerformance(perf, set="Training"): 
    print(f'\033[1m {set} set performance \033[0m')
    print("----------------------------------------")
    print(f'Accuracy: {perf["accuracy"]: .4f}')
    print(f'Precision: {perf["precision"]: .4f}')
    print(f'Recall: {perf["recall"]: .4f}')
    print(f'F1 score: {perf["f1"]: .4f}')
    print('\n')

# --- Print the performance
def print_performances(y_true, y_pred, set="Training"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(set)
    print(f'Accuracy: {accuracy: .4f}')
    print(f'Precision: {precision: .4f}')
    print(f'Recall: {recall: .4f}')
    print(f'F1 score: {f1: .4f}')
    print(" ")


# --- Print for Latex tables
def print_latex(y_train, y_train_pred, y_test, y_test_pred):
    s1 = f"{f1_score(y_train, y_train_pred) * 100:.2f} \% &"
    s2 = f" {precision_score(y_train, y_train_pred) * 100:.2f} \% &"
    s3 = f" {recall_score(y_train, y_train_pred) * 100:.2f} \% &"

    s4 = f" {f1_score(y_test, y_test_pred) * 100:.2f} \% &"
    s5 = f" {precision_score(y_test, y_test_pred) * 100:.2f} \% &"
    s6 = f" {recall_score(y_test, y_test_pred) * 100:.2f} \%"

    s = s1 + s2 + s3 + s4 + s5 + s6
    print(s)



#--------------------
# --- Autoencoder ---
# -------------------
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(25, activation="relu"),
            layers.Dense(14, activation="relu")
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(25, activation="relu"),
            layers.Dense(50, activation="linear")
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def AE(normal_train_data, test_data, n_std, seed):
    
    # --- Random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # --- Autoencoder training
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss='mae')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(normal_train_data, normal_train_data,
                    epochs=500,
                    batch_size=64,
                    validation_data=(test_data, test_data),
                    shuffle=True,
                    callbacks=[early_stopping],
                    verbose=0
                    )
    
    # --- Cut off value
    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    cut_off = np.mean(train_loss) + n_std * np.std(train_loss)

    return autoencoder, cut_off


def predict_AE(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)


def find_best_threshold(model, train_data, train_labels):

    # --- Cut-off values to explore
    reconstructions = model.predict(train_data)
    train_loss = tf.keras.losses.mae(reconstructions, train_data)
    thresholds = tf.linspace(tf.reduce_min(train_loss), tf.reduce_max(train_loss), 1000)
    best_threshold = None
    best_f1 = -1
    
    for threshold in thresholds:
        y_pred = predict_AE(model, train_data, threshold)
        score = f1_score(train_labels, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    
    return best_threshold