import numpy as np
from scipy.linalg import eigh

# Step 1: Feature Extraction and Data Organization
def extract_features(eeg_data, frequency_bands):
    """
    Extract Power Spectral Density (PSD) features from EEG data.
    
    Parameters:
    eeg_data (ndarray): EEG data of shape (subjects, electrodes, time_points)
    frequency_bands (list): List of frequency bands (e.g., ['theta', 'alpha', 'beta'])
    
    Returns:
    X_m (ndarray): Data matrix for each subject of shape (electrodes * frequency_bands, time_points)
    """
    subjects, electrodes, time_points = eeg_data.shape
    num_freq_bands = len(frequency_bands)
    
    # Initialize the data matrix X_m for each subject
    X_m = np.zeros((subjects, electrodes * num_freq_bands, time_points))
    
    for m in range(subjects):
        for d in range(electrodes):
            for f, freq_band in enumerate(frequency_bands):
                # Calculate PSD for each electrode and frequency band (Formula 1)
                psd = np.abs(np.fft.fft(eeg_data[m, d, :])) ** 2
                X_m[m, d * num_freq_bands + f, :] = psd
    
    return X_m

# Step 2: CorrCA Computation
def corrca(X_m):
    """
    Perform Correlated Component Analysis (CorrCA) on the data matrix X_m.
    
    Parameters:
    X_m (ndarray): Data matrix for each subject of shape (subjects, electrodes * frequency_bands, time_points)
    
    Returns:
    W (ndarray): Linear combination weight matrix of shape (electrodes * frequency_bands, components)
    eigenvalues (ndarray): Eigenvalues corresponding to the components
    """
    subjects, features, time_points = X_m.shape
    
    # Compute the within-subject covariance matrix (Σwithin)
    Σwithin = np.zeros((features, features))
    for m in range(subjects):
        X_m_centered = X_m[m] - np.mean(X_m[m], axis=1, keepdims=True)
        Σwithin += np.cov(X_m_centered)
    Σwithin /= subjects
    
    # Compute the between-subject covariance matrix (Σbetween)
    X_mean = np.mean(X_m, axis=0)
    X_mean_centered = X_mean - np.mean(X_mean, axis=1, keepdims=True)
    Σbetween = np.cov(X_mean_centered)
    
    # Solve the generalized eigenvalue problem (Formula 3)
    eigenvalues, W = eigh(Σbetween, Σwithin)
    
    return W, eigenvalues

# Step 3: Component Selection and Dimensionality Reduction
def reduce_dimensions(X_m, W, num_components=3):
    """
    Reduce the dimensionality of the data using the top components.
    
    Parameters:
    X_m (ndarray): Data matrix for each subject of shape (subjects, electrodes * frequency_bands, time_points)
    W (ndarray): Linear combination weight matrix of shape (electrodes * frequency_bands, components)
    num_components (int): Number of components to retain
    
    Returns:
    Y_mk (ndarray): Reduced components of shape (subjects, num_components, time_points)
    """
    subjects, features, time_points = X_m.shape
    
    # Select the top components (Formula 4)
    W_top = W[:, -num_components:]
    
    # Transform the data to the reduced component space
    Y_mk = np.zeros((subjects, num_components, time_points))
    for m in range(subjects):
        Y_mk[m] = np.dot(W_top.T, X_m[m])
    
    return Y_mk

def corrca_fit(X, gamma=0, k=None):
    N, D, T = X.shape
    if k is None:
        k = D
    Rw = np.zeros((D, D))
    for n in range(N):
        Rw += np.cov(X[n, :, :])
    Rw = Rw / N
    Rt = N**2 * np.cov(np.mean(X, axis=0))
    Rb = (Rt - Rw) / (N - 1)
    rank = np.linalg.matrix_rank(Rw)
    k = min(k, rank)
    if k < D:
        U, S, Vh = svd(Rw)
        invR = U[:, :k] @ np.diag(1.0 / S[:k]) @ Vh[:k, :]
        eigvals, eigvecs = eig(invR @ Rb)
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]
    else:
        Rw_reg = (1 - gamma) * Rw + gamma * np.mean(np.diag(Rw)) * np.eye(D)
        eigvals, eigvecs = eig(Rb, Rw_reg)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    ISC = np.diag(eigvecs.T @ Rb @ eigvecs) / np.diag(eigvecs.T @ Rw @ eigvecs)
    if k < D:
        A = Rw @ eigvecs @ inv(eigvecs.T @ Rw @ eigvecs)
    else:
        A = Rw @ eigvecs @ np.diag(1 / np.diag(eigvecs.T @ Rw @ eigvecs))
    return eigvecs, ISC, A

def corrca_transform(X, W):
    N, D, T = X.shape
    K = W.shape[1]
    Y = np.zeros((N, K, T))
    for n in range(N):
        Y[n, :, :] = W.T @ X[n, :, :]
    return Y

data = np.load('cleaned_eeg_data.npy', allow_pickle=True)
X_all = np.mean(data, axis=1)
W, ISC, A = corrca_fit(X_all, gamma=0.1, k=3)
Y = corrca_transform(X_all, W)
np.save('corrca_W.npy', W)
np.save('corrca_ISC.npy', ISC)
np.save('corrca_A.npy', A)
np.save('corrca_Y.npy', Y)
plt.figure(figsize=(8, 6))
plt.plot(ISC, 'o-', linewidth=2)
plt.xlabel('Component Number')
plt.ylabel('Inter-Subject Correlation (ISC)')
plt.title('CorrCA ISC Spectrum')
plt.show()

if __name__ == "__main__":
    # Example EEG data (subjects, electrodes, time_points)
    eeg_data = np.random.rand(10, 32, 1000)  # 10 subjects, 32 electrodes, 1000 time points
    frequency_bands = ['theta', 'alpha', 'beta']
    
    # Step 1: Feature Extraction
    X_m = extract_features(eeg_data, frequency_bands)
    
    # Step 2: CorrCA Computation
    W, eigenvalues = corrca(X_m)
    
    # Step 3: Dimensionality Reduction
    Y_mk = reduce_dimensions(X_m, W, num_components=3)
    
    print("Reduced components shape:", Y_mk.shape)
