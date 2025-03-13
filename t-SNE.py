import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def standardize_features(X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std

def apply_tsne(X_std, perplexity=30, learning_rate=200):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    Y = tsne.fit_transform(X_std)
    return Y

def visualize_tsne(Y, labels):
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(Y[labels == label, 0], Y[labels == label, 1], label=f'Condition {label}')
    plt.title('t-SNE Visualization of EEG Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

def calculate_jsd(P, Q):
    M = 0.5 * (P + Q)
    kl_PM = np.sum(P * np.log(P / M))
    kl_QM = np.sum(Q * np.log(Q / M))
    jsd = 0.5 * (kl_PM + kl_QM)
    return jsd

np.random.seed(42)
n_neurons = 50   
n_time_steps = 1000 
neural_data = np.random.rand

for i in range(0, n_time_steps, 200):
    neural_data[i:i+100, :] += np.sin(np.linspace(0, np.pi, n_neurons))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_2d = tsne.fit_transform(neural_data)

for lr in [10, 50, 200, 500, 750, 1000]:
    tsne = TSNE(n_components=2, learning_rate=lr, random_state=42)
    data_2d = tsne.fit_transform(neural_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=np.arange(len(data_2d)), cmap=plt.cm.Spectral)
    plt.title(f"t-SNE with Learning Rate={lr}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(label="Time Index")
    plt.show()
