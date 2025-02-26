import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import seaborn as sns

# Calculate EMD
def calculate_emd(real_data, synthetic_data):
    emd_scores = []
    for i in range(real_data.shape[1]):
        if np.issubdtype(real_data[:, i].dtype, np.number):  # Only calculate EMD for numerical columns
            emd = wasserstein_distance(real_data[:, i], synthetic_data[:, i])
            emd_scores.append(emd)
    return np.mean(emd_scores)


# Calculate KS-test
def calculate_ks_test(real_data, synthetic_data):
    ks_stats = {}
    for i in range(real_data.shape[1]):
        ks_stat = ks_2samp(real_data[:, i], synthetic_data[:, i]).statistic
        ks_stats[f'Feature {i}'] = ks_stat
    return ks_stats

from scipy.spatial.distance import jensenshannon

# Calculate Jensen-Shannon Divergence
def calculate_js_divergence(real_data, synthetic_data, num_bins=10):
    js_divs = {}
    for i in range(real_data.shape[1]):
        real_counts, _ = np.histogram(real_data[:, i], bins=num_bins, density=True)
        synthetic_counts, _ = np.histogram(synthetic_data[:, i], bins=num_bins, density=True)
        
        # Normalize the counts
        real_counts = real_counts / np.sum(real_counts)
        synthetic_counts = synthetic_counts / np.sum(synthetic_counts)
        
        js_div = jensenshannon(real_counts, synthetic_counts)
        js_divs[f'Feature {i}'] = js_div
    return js_divs

# Calculate S_basic
def calculate_sbasic(X_real, X_synthetic):
    real_means = np.mean(X_real, axis=0)
    real_stds = np.std(X_real, axis=0)
    synthetic_means = np.mean(X_synthetic, axis=0)
    synthetic_stds = np.std(X_synthetic, axis=0)

    means_corr, _ = spearmanr(real_means, synthetic_means)
    stds_corr, _ = spearmanr(real_stds, synthetic_stds)

    return (means_corr + stds_corr) / 2

# Calculate S_correlation
def calculate_scorr(X_real, X_synthetic):
    real_corr_matrix = np.corrcoef(X_real, rowvar=False)
    synthetic_corr_matrix = np.corrcoef(X_synthetic, rowvar=False)
    
    corr_flattened_real = real_corr_matrix.flatten()
    corr_flattened_synthetic = synthetic_corr_matrix.flatten()
    
    scorr, _ = spearmanr(corr_flattened_real, corr_flattened_synthetic)
    
    return scorr

# Calculate S_Mirror
def calculate_smirr(X_real, X_synthetic):
    num_features = X_real.shape[1]
    associations = []

    for i in range(num_features):
        column_real = X_real[:, i]
        column_synthetic = X_synthetic[:, i]
        correlation, _ = spearmanr(column_real, column_synthetic)
        associations.append(correlation)
    
    smirr = np.mean(associations)
    return smirr

# Calculate S_PCA
def calculate_spca(X_real, X_synthetic, n_components=2, epsilon=1e-10):
    """
    Compute the PCA similarity score (Spca) between the explained variances of the 
    principal components of the real and synthetic datasets.
    
    Parameters:
    X_real (numpy.ndarray): The real dataset.
    X_synthetic (numpy.ndarray): The synthetic dataset.
    n_components (int): Number of principal components to consider.
    epsilon (float): Small value to avoid division by zero.
    
    Returns:
    float: The Spca score.
    """
    # Perform PCA on real data
    pca_real = PCA(n_components=n_components)
    pca_real.fit(X_real)
    explained_variance_real = pca_real.explained_variance_ratio_

    # Perform PCA on synthetic data
    pca_synthetic = PCA(n_components=n_components)
    pca_synthetic.fit(X_synthetic)
    explained_variance_synthetic = pca_synthetic.explained_variance_ratio_

    # Apply logarithm to the explained variances
    log_explained_variance_real = np.log(explained_variance_real + epsilon)
    log_explained_variance_synthetic = np.log(explained_variance_synthetic + epsilon)

    # Compute MAPE between the log-transformed explained variances
    mape = np.mean(np.abs((log_explained_variance_real - log_explained_variance_synthetic) / log_explained_variance_real)) * 100

    # Calculate the similarity score
    spca = 1 - (mape / 100)

    return spca