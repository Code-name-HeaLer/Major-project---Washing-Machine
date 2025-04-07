# --- START OF FILE graph.py ---

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import networkx as nx
import math
import random
from collections import Counter
from scipy.stats import skew, kurtosis
import traceback

print("\n--- Graph Construction Setup ---")

# --- Parameters ---
FEATURE_THRESHOLD = 10.0  # Threshold for 'active' points in feature extraction
MAX_K_TO_TEST = 15       # Maximum number of clusters to evaluate
RANDOM_SEED = 42
# <<< IMPORTANT: This needs to be determined after analyzing the training data features >>>
# <<< Using 9 as a placeholder based on graph_2.py analysis, but ideally determined dynamically or via user input >>>
CHOSEN_N_CLUSTERS = 9
N_SAMPLES_PLOT_PER_CLUSTER = 3 # Number of example windows to plot for each cluster

# Set random seeds for reproducibility in clustering
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
# Note: KMeans uses numpy's random state, which is seeded.

# --- Feature Extraction Function (from graph_2.py, verified OK) ---
def extract_features(window, threshold=FEATURE_THRESHOLD):
    """
    Extracts a 15-dimensional feature vector from a single time-series window.

    Args:
        window (np.ndarray): 1D array representing the time-series window.
        threshold (float): Threshold to determine 'active' points.

    Returns:
        np.ndarray: 15-dimensional feature vector.
    """
    features = []
    epsilon = 1e-9 # To avoid division by zero or issues with zero std dev

    # Basic Stats
    mean_val = np.mean(window)
    std_val = np.std(window)
    features.append(mean_val)
    features.append(std_val)
    features.append(np.max(window))
    features.append(np.min(window))
    features.append(np.sqrt(np.mean(window**2))) # RMS

    # Shape Stats (handle low std dev)
    features.append(skew(window) if std_val > epsilon else 0)
    features.append(kurtosis(window) if std_val > epsilon else 0)

    # Active Power Stats
    active_indices = np.where(window > threshold)[0]
    n_active = len(active_indices)
    features.append(n_active)

    if n_active > 0:
        active_signal = window[active_indices]
        features.append(np.mean(active_signal))
        features.append(np.sum(active_signal)) # Energy approximation
        features.append(np.max(active_signal))
        # Peak count in active signal (simple difference method)
        if len(active_signal) > 2:
            diffs = np.diff(active_signal)
            peaks = np.sum((diffs[:-1] > 0) & (diffs[1:] < 0))
            features.append(peaks)
        else:
            features.append(0) # Not enough points for peaks
    else:
        # Append zeros if no active power
        features.extend([0, 0, 0, 0])

    # FFT Features (handle potential errors)
    try:
        n_fft = len(window)
        if n_fft > 1: # Need at least 2 points for FFT
             fft_vals = np.fft.rfft(window) # Real FFT for real signal
             fft_mag_sq = np.abs(fft_vals)**2 # Power spectrum
             n_rfft = len(fft_mag_sq)

             # Define frequency bins (adjust as needed)
             # Bin 1: Low frequencies (e.g., first 10%)
             # Bin 2: Mid frequencies (e.g., 10% to 50%)
             # Bin 3: High frequencies (e.g., 50% to end)
             bin1_end = max(1, n_rfft // 10) # Ensure at least one element
             bin2_end = n_rfft // 2

             energy_bin1 = np.sum(fft_mag_sq[0:bin1_end])
             energy_bin2 = np.sum(fft_mag_sq[bin1_end:bin2_end])
             energy_bin3 = np.sum(fft_mag_sq[bin2_end:])
             total_energy = energy_bin1 + energy_bin2 + energy_bin3 + epsilon # Avoid division by zero

             features.extend([energy_bin1 / total_energy,
                              energy_bin2 / total_energy,
                              energy_bin3 / total_energy])
        else:
             features.extend([0, 0, 0]) # Not enough points for FFT
    except Exception as fft_e:
        print(f"Warning: FFT error - {fft_e}. Appending zeros for FFT features.")
        features.extend([0, 0, 0])

    # Ensure features are finite and numeric, handle NaNs/Infs
    final_features = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Double check after nan_to_num (paranoid check)
    if np.any(np.isinf(final_features)) or np.any(np.isnan(final_features)):
        print("Warning: NaNs or Infs still detected after nan_to_num. Applying again.")
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure correct dimension (should be 15)
    if len(final_features) != 15:
         print(f"Warning: Feature vector length is {len(final_features)}, expected 15. Padding/truncating if necessary.")
         # This part needs careful handling based on why the length might be wrong
         # For now, let's assume it's correct or raise an error if critical
         if len(final_features) < 15:
              final_features = np.pad(final_features, (0, 15 - len(final_features)), 'constant', constant_values=0.0)
         elif len(final_features) > 15:
              final_features = final_features[:15]


    return final_features

# --- Cluster Analysis Function (from graph_2.py, verified OK) ---
def analyze_clusters_features(scaled_train_features, max_k=MAX_K_TO_TEST, plot=True):
    """
    Analyzes the optimal number of clusters (K) for feature vectors using
    the Elbow method (inertia) and Silhouette score. Operates only on training data features.

    Args:
        scaled_train_features (np.ndarray): Scaled feature vectors for the training set.
        max_k (int): Maximum K value to test.
        plot (bool): Whether to display the analysis plots.

    Returns:
        int: Suggested K based on analysis (e.g., highest silhouette score), or CHOSEN_N_CLUSTERS if analysis fails.
               Returns None if clustering cannot be performed.
    """
    print(f"\n--- Analyzing Optimal K for Features (Train Data, up to K={max_k}) ---")
    n_train_samples = scaled_train_features.shape[0]

    if scaled_train_features.ndim != 2 or n_train_samples < 2:
        print("  Warning: Insufficient training features for cluster analysis. Cannot determine optimal K.")
        return None

    # K must be less than the number of samples
    actual_max_k = min(max_k, n_train_samples - 1)

    if actual_max_k < 2:
        print(f"  Warning: Only {n_train_samples} training samples available. Cannot perform clustering analysis (K must be >= 2).")
        return None

    k_range = range(2, actual_max_k + 1)
    print(f"  Testing K values on {n_train_samples} training features: {list(k_range)}")

    inertia = []
    silhouette_avg = []
    best_k_silhouette = -1
    max_silhouette = -1 # Silhouette score is between -1 and 1

    for k in k_range:
        current_inertia = np.nan
        current_silhouette = np.nan
        try:
            # Ensure n_init is appropriate, 'auto' might be better in newer sklearn
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, init='k-means++')
            cluster_labels = kmeans.fit_predict(scaled_train_features)
            current_inertia = kmeans.inertia_

            # Silhouette score requires at least 2 unique clusters
            n_unique_labels = len(np.unique(cluster_labels))
            if n_unique_labels > 1:
                current_silhouette = silhouette_score(scaled_train_features, cluster_labels)
                if current_silhouette > max_silhouette:
                    max_silhouette = current_silhouette
                    best_k_silhouette = k
            else:
                 print(f"    Warning: Only {n_unique_labels} cluster found for K={k}. Cannot compute Silhouette score.")

        except Exception as e:
            print(f"    Error during clustering analysis for K={k}: {e}")

        inertia.append(current_inertia)
        silhouette_avg.append(current_silhouette)

    if plot:
        plt.figure(figsize=(14, 6))
        valid_inertia_idx = ~np.isnan(inertia)
        valid_silhouette_idx = ~np.isnan(silhouette_avg)

        plt.subplot(1, 2, 1)
        if np.any(valid_inertia_idx):
            plt.plot(np.array(list(k_range))[valid_inertia_idx], np.array(inertia)[valid_inertia_idx], marker='o')
            plt.xticks(np.array(list(k_range))[valid_inertia_idx])
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Inertia (WCSS)')
            plt.title('Elbow Method (Training Features)')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "Inertia values not available", ha='center', va='center')

        plt.subplot(1, 2, 2)
        if np.any(valid_silhouette_idx):
            plt.plot(np.array(list(k_range))[valid_silhouette_idx], np.array(silhouette_avg)[valid_silhouette_idx], marker='o')
            plt.xticks(np.array(list(k_range))[valid_silhouette_idx])
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Average Silhouette Score')
            plt.title('Silhouette Analysis (Training Features)')
            plt.grid(True)
            if best_k_silhouette > 0:
                 plt.axvline(x=best_k_silhouette, color='red', linestyle='--', label=f'Best K (Silhouette) = {best_k_silhouette}')
                 plt.legend()
        else:
             plt.text(0.5, 0.5, "Silhouette scores not available", ha='center', va='center')


        plt.tight_layout()
        plt.show()

    print("--- Feature Cluster Analysis (Train Data) Done ---")

    if best_k_silhouette > 0:
         print(f"Suggested K based on highest Silhouette score: {best_k_silhouette} (Score: {max_silhouette:.4f})")
         return best_k_silhouette
    else:
         print(f"Could not determine a suggested K. Using default: {CHOSEN_N_CLUSTERS}")
         return CHOSEN_N_CLUSTERS


# --- Clustering Function (from graph_2.py, verified OK) ---
def cluster_features_and_get_centroids(scaled_features, n_clusters, train_indices):
    """
    Performs KMeans clustering on the scaled training features and returns the
    fitted model, centroids, and labels for *all* samples.

    Args:
        scaled_features (np.ndarray): Scaled feature vectors for *all* samples.
        n_clusters (int): The chosen number of clusters (K).
        train_indices (list): Indices corresponding to the training samples.

    Returns:
        tuple: (kmeans_model, all_labels, feature_centroids)
               - KMeans: The fitted KMeans model object.
               - np.ndarray: Cluster labels assigned to *all* samples.
               - np.ndarray: Coordinates of the cluster centroids [n_clusters, n_features].
    """
    print(f"\n--- Clustering Features into {n_clusters} Clusters ---")
    scaled_train_features = scaled_features[train_indices]
    n_train_samples = scaled_train_features.shape[0]

    if n_train_samples < n_clusters:
        print(f"Warning: Number of training samples ({n_train_samples}) is less than n_clusters ({n_clusters}). Reducing n_clusters to {n_train_samples}.")
        n_clusters = n_train_samples
        if n_clusters <= 1:
             raise ValueError("Cannot perform clustering with <= 1 training sample/cluster.")


    print(f"  Fitting KMeans on {n_train_samples} training features (K={n_clusters})...")
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10, init='k-means++')
    train_labels = kmeans_model.fit_predict(scaled_train_features)
    feature_centroids = kmeans_model.cluster_centers_
    print("  KMeans fitting complete.")

    # Verify clustering results on training data
    train_cluster_counts = dict(Counter(train_labels))
    n_found_train = len(train_cluster_counts)
    print(f"  Training cluster label counts: {train_cluster_counts}")
    print(f"  Found {n_found_train} non-empty clusters in training data.")
    if n_found_train < n_clusters:
        print(f"  Warning: Only found {n_found_train} clusters in training data, although K={n_clusters} was requested. Some centroids might represent empty clusters initially.")


    # Predict labels for all samples using the fitted model
    print(f"  Predicting cluster labels for all {scaled_features.shape[0]} samples...")
    all_labels = kmeans_model.predict(scaled_features)
    print(f"  All labels shape: {all_labels.shape}")
    print(f"  Overall cluster label counts: {dict(Counter(all_labels))}")

    print(f"  Node features (Feature Centroids) shape: {feature_centroids.shape}") # [n_clusters, n_features=15]
    print("--- Feature Clustering Done ---")
    return kmeans_model, all_labels, feature_centroids

# --- Transition Matrix Function (from graph_2.py, verified OK) ---
def calculate_transition_matrix(all_labels, train_indices, n_clusters):
    """
    Calculates the transition probability matrix based *only* on the sequence of
    cluster labels observed in the training data.

    Args:
        all_labels (np.ndarray): Cluster labels assigned to *all* samples.
        train_indices (list): Indices corresponding to the training samples.
        n_clusters (int): The total number of clusters (K).

    Returns:
        np.ndarray: Transition probability matrix [n_clusters, n_clusters],
                    where T[i, j] is the probability of transitioning from cluster i to j.
    """
    print("\n--- Calculating Transition Matrix (Based on Training Data Labels) ---")
    # Extract the sequence of labels corresponding to the training data order
    # IMPORTANT: This assumes train_indices preserves the temporal order within the training set.
    # If train_indices are shuffled, this calculation might not represent true temporal transitions.
    # For now, we proceed assuming train_indices reflects a meaningful sequence for transitions.
    train_labels_sequence = all_labels[train_indices]
    num_windows = len(train_labels_sequence)

    if num_windows < 2:
        print("  Warning: Need at least 2 training labels to calculate transitions. Returning identity matrix.")
        # Return identity matrix, implying no transitions or self-loops only
        return np.eye(n_clusters, dtype=float)

    transition_counts = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(num_windows - 1):
        current_state = train_labels_sequence[i]
        next_state = train_labels_sequence[i+1]

        # Ensure labels are within the valid range (0 to n_clusters-1)
        if 0 <= current_state < n_clusters and 0 <= next_state < n_clusters:
            transition_counts[current_state, next_state] += 1
        else:
            print(f"  Warning: Encountered invalid state label during transition calculation: {current_state} -> {next_state}")

    # Apply Laplace Smoothing (add-1 smoothing) to avoid zero probabilities
    transition_counts += 1

    # Normalize counts to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)

    # Handle cases where a state might have no outgoing transitions (even with smoothing, shouldn't happen)
    transition_probabilities = np.divide(transition_counts, row_sums,
                                         out=np.zeros_like(transition_counts, dtype=float),
                                         where=row_sums!=0)

    # For rows with sum 0 (shouldn't occur with smoothing), set self-loop probability to 1
    zero_sum_rows = np.where(row_sums == 0)[0]
    if len(zero_sum_rows) > 0:
        print(f"  Warning: Rows {zero_sum_rows} had zero sum even after smoothing. Setting self-loop prob to 1.")
        for idx in zero_sum_rows:
             if 0 <= idx < n_clusters:
                transition_probabilities[idx, idx] = 1.0


    print(f"  Transition Probability Matrix shape: {transition_probabilities.shape}")
    print("--- Transition Matrix Calculation Done ---")
    return transition_probabilities

# --- Function to Visualize Window Assignments ---
def plot_window_assignments(all_signals_padded, all_metadata_with_labels, n_clusters, n_samples_per_cluster=3):
    """
    Plots a few sample time-series windows assigned to each cluster (node).

    Args:
        all_signals_padded (np.ndarray): Padded signals for all samples [N, seq_len].
        all_metadata_with_labels (list): Metadata list updated with 'cluster_label'.
        n_clusters (int): The total number of clusters (nodes).
        n_samples_per_cluster (int): Number of sample windows to plot per cluster.
    """
    print(f"\n--- Visualizing Window Assignments to Clusters (Nodes) ---")
    if n_clusters <= 0:
        print("  Cannot plot assignments: 0 clusters.")
        return

    seq_len = all_signals_padded.shape[1]
    time_steps = np.arange(seq_len)

    # Determine grid layout
    ncols = math.ceil(math.sqrt(n_clusters))
    nrows = math.ceil(n_clusters / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten for easy indexing

    # Determine global y-limits for consistent scaling
    global_min = np.min(all_signals_padded)
    global_max = np.max(all_signals_padded)
    y_padding = (global_max - global_min) * 0.1
    y_min = global_min - y_padding
    y_max = global_max + y_padding

    plotted_clusters = 0
    for cluster_label in range(n_clusters):
        ax = axes[cluster_label]
        indices_in_cluster = [i for i, meta in enumerate(all_metadata_with_labels) if meta.get('cluster_label') == cluster_label]

        if not indices_in_cluster:
            ax.text(0.5, 0.5, f'Node {cluster_label}\n(No Samples)', ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            plotted_clusters += 1
            # Sample indices from this cluster
            n_to_sample = min(n_samples_per_cluster, len(indices_in_cluster))
            sampled_global_indices = random.sample(indices_in_cluster, n_to_sample)

            # Plot sampled windows
            for global_idx in sampled_global_indices:
                signal = all_signals_padded[global_idx, :]
                # Use alpha for potentially overlapping lines
                ax.plot(time_steps, signal, alpha=0.7, linewidth=1.0, label=f'Win {global_idx}')

            ax.set_title(f"Node {cluster_label} ({len(indices_in_cluster)} windows)", fontsize=10)
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            # Add labels only to edge plots
            if cluster_label >= n_clusters - ncols : ax.set_xlabel("Time Step")
            if cluster_label % ncols == 0: ax.set_ylabel("Signal Value")
            ax.tick_params(axis='both', which='major', labelsize=8)
            # Optional: Add legend if n_samples_per_cluster is small
            # if n_to_sample <= 5: ax.legend(fontsize=6)

    # Hide unused subplots
    for j in range(n_clusters, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Sample Windows Assigned to each of the {plotted_clusters}/{n_clusters} Populated Nodes", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("--- Window Assignment Visualization Done ---")

# --- Graph Creation/Visualization Function (from graph_2.py, verified OK, adjusted node features) ---
def create_and_visualize_graph(transition_matrix, node_features_centroids, n_clusters, visualize=True):
    """
    Creates a NetworkX directed graph from the transition matrix and node features (centroids).
    Optionally visualizes the graph.

    Args:
        transition_matrix (np.ndarray): Transition probability matrix [n_clusters, n_clusters].
        node_features_centroids (np.ndarray): Feature centroids [n_clusters, n_features].
        n_clusters (int): The number of nodes (clusters).
        visualize (bool): Whether to plot the graph.

    Returns:
        networkx.DiGraph: The constructed graph with node features and edge weights.
                          Returns None if graph creation fails.
    """
    print("\n--- Creating Graph Structure ---")
    if n_clusters == 0:
        print("  Error: Cannot create graph with 0 nodes.")
        return None
    if transition_matrix.shape[0] != n_clusters or transition_matrix.shape[1] != n_clusters:
        print(f"  Error: Transition matrix shape {transition_matrix.shape} incompatible with n_clusters {n_clusters}.")
        return None
    if node_features_centroids.shape[0] != n_clusters:
        print(f"  Error: Node feature centroids shape {node_features_centroids.shape} incompatible with n_clusters {n_clusters}.")
        return None

    G = nx.DiGraph()
    # Add nodes with features
    for node_idx in range(n_clusters):
        # Node features are the corresponding centroid vectors
        features = np.array(node_features_centroids[node_idx, :], dtype=float)
        G.add_node(node_idx, features=features, label=f'Node {node_idx}') # Use index as label for simplicity

    print(f"  Added {G.number_of_nodes()} nodes. Node labels are indices 0 to {n_clusters-1}.")
    print(f"  Node feature vector length: {node_features_centroids.shape[1]}") # Should be 15

    # Add edges with weights (transition probabilities)
    edges_added = 0
    edge_weights_dict = {}
    min_weight_threshold = 1e-6 # Threshold to avoid adding negligible edges

    for i in range(n_clusters):
        for j in range(n_clusters):
            weight = transition_matrix[i, j]
            if weight > min_weight_threshold:
                G.add_edge(i, j, weight=float(weight))
                edge_weights_dict[(i, j)] = f"{weight:.2f}"
                edges_added += 1

    print(f"  Added {edges_added} weighted edges based on training transitions (weight > {min_weight_threshold}).")

    if visualize and G.number_of_nodes() > 0:
        print("  Visualizing graph...")
        plt.figure(figsize=(max(10, n_clusters * 1.2), max(9, n_clusters * 1.1)))
        # Use a layout that suits directed graphs
        pos = nx.spring_layout(G, k=1.8/math.sqrt(n_clusters) if n_clusters>1 else 1, iterations=70, seed=RANDOM_SEED) if G.number_of_edges() > 0 else nx.circular_layout(G)

        node_draw_labels = {idx: data['label'] for idx, data in G.nodes(data=True)}

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_draw_labels, font_size=10, font_weight='bold')

        if G.number_of_edges() > 0:
            edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
            max_w = max(edge_weights_list) if edge_weights_list else 1
            max_w = max(max_w, min_weight_threshold) # Avoid division by zero if all weights are tiny

            # Scale edge width and alpha based on weight
            edge_widths = [(w / max_w * 3.5) + 0.5 for w in edge_weights_list]
            edge_alphas = [(w / max_w * 0.7) + 0.2 for w in edge_weights_list]

            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=edge_alphas, edge_color='gray',
                                   arrowstyle='-|>', arrowsize=18, node_size=800)

            # Optionally draw edge labels (can get cluttered for many edges)
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights_dict, label_pos=0.4,
            #                              font_size=8, font_color='darkred',
            #                              bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
        else:
            print("  No edges to draw.")

        plt.title(f"Graph Structure ({n_clusters} Nodes based on Feature Clusters)", fontsize=14)
        plt.axis('off')
        plt.show()
        print("--- Graph Visualization Done ---")
    elif visualize:
         print("  Cannot visualize empty graph.")


    print("--- Graph Creation Done ---")
    return G

# --- Main Orchestration Function ---
def construct_graph_nodes_and_adjacency(all_signals_padded, all_metadata, train_indices,
                                        n_clusters_input=CHOSEN_N_CLUSTERS,
                                        perform_k_analysis=False, # Set to True to run elbow/silhouette plots
                                        visualize_graph=True,visualize_assignments=True):
    """
    Constructs the graph by extracting features, clustering, calculating transitions,
    and building the NetworkX graph object.

    Args:
        all_signals_padded (np.ndarray): Padded signals for all samples [N, seq_len].
        all_metadata (list): Metadata list for all samples.
        train_indices (list): List of global indices for the training set.
        n_clusters_input (int): The desired number of clusters (K). Can be overridden by analysis.
        perform_k_analysis (bool): If True, run silhouette/elbow analysis and plot.
        visualize_graph (bool): If True, plot the final graph structure.

    Returns:
        tuple: (G, node_features_centroids, all_metadata_with_labels)
               - networkx.DiGraph: The constructed graph.
               - np.ndarray: Feature centroids array [n_clusters, n_features].
               - list: Original metadata list updated with 'cluster_label' for each sample.
               Returns (None, None, all_metadata) if graph construction fails.
    """
    print("\n=== Starting Graph Construction Process ===")
    n_total_samples = all_signals_padded.shape[0]
    n_features = 15 # Based on extract_features function

    # 1. Extract Features for ALL samples
    print(f"\n--- Extracting {n_features} Features for {n_total_samples} Windows ---")
    all_feature_vectors = np.array([extract_features(window) for window in all_signals_padded])
    print(f"  Shape of feature vectors (all): {all_feature_vectors.shape}")

    # 2. Scale Features (Fit ONLY on training features)
    print("\n--- Scaling Features ---")
    train_features = all_feature_vectors[train_indices]
    if train_features.shape[0] == 0:
         raise ValueError("Cannot scale features: No training data provided.")

    feature_scaler = StandardScaler().fit(train_features)
    scaled_all_features = feature_scaler.transform(all_feature_vectors)
    # Handle potential NaNs/Infs after scaling
    scaled_all_features = np.nan_to_num(scaled_all_features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Shape of scaled features (all): {scaled_all_features.shape}")
    scaled_train_features = scaled_all_features[train_indices] # Get scaled training features

    # 3. Analyze K (Optional, on scaled training features)
    chosen_k = n_clusters_input
    if perform_k_analysis:
        suggested_k = analyze_clusters_features(scaled_train_features, max_k=MAX_K_TO_TEST, plot=True)
        if suggested_k is not None:
             print(f"  K analysis suggests K={suggested_k}. Using this value.")
             chosen_k = suggested_k
        else:
             print(f"  K analysis failed. Using the provided n_clusters_input: {n_clusters_input}")
             chosen_k = n_clusters_input
    else:
         print(f"\n--- Skipping K Analysis ---")
         print(f"  Using predefined n_clusters: {chosen_k}")


    if chosen_k <= 1:
         print(f"Error: Invalid number of clusters selected: {chosen_k}. Must be > 1.")
         return None, None, all_metadata


    # 4. Cluster Features (Fit on Train, Predict on All) & Get Centroids
    try:
        kmeans_model, all_labels, node_features_centroids = cluster_features_and_get_centroids(
            scaled_all_features, chosen_k, train_indices
        )
        actual_n_clusters = node_features_centroids.shape[0] # Number of centroids returned
        if actual_n_clusters != chosen_k:
             print(f"Warning: KMeans returned {actual_n_clusters} centroids, but requested K was {chosen_k}.")
             # Potentially adjust chosen_k if needed, though usually centroids match K.
             chosen_k = actual_n_clusters
             if chosen_k <=1 : raise ValueError("Clustering resulted in <=1 cluster.")

    except ValueError as ve:
        print(f"Error during clustering: {ve}")
        return None, None, all_metadata

    # 5. Add cluster labels to metadata
    if len(all_labels) != len(all_metadata):
        print(f"Error: Mismatch between number of labels ({len(all_labels)}) and metadata entries ({len(all_metadata)}).")
        return None, None, all_metadata
    for i in range(len(all_metadata)):
        all_metadata[i]['cluster_label'] = all_labels[i] # Store the node index/cluster label

    # 5b. Visualize window assignments (Optional)
    if visualize_assignments:
        plot_window_assignments(all_signals_padded, all_metadata, chosen_k, N_SAMPLES_PLOT_PER_CLUSTER)  

    # 6. Calculate Transition Matrix (Based on Training Labels sequence)
    transition_matrix = calculate_transition_matrix(all_labels, train_indices, chosen_k)

    # 7. Create Graph
    graph = create_and_visualize_graph(transition_matrix, node_features_centroids, chosen_k, visualize=visualize_graph)

    if graph is None:
        print("Error: Graph creation failed.")
        return None, None, all_metadata

    print("\n=== Graph Construction Process Finished Successfully ===")
    return graph, node_features_centroids, all_metadata


# --- Example Usage (Illustrative - requires data from dataset.py) ---
if __name__ == "__main__":
    print("\n--- Running graph.py directly for testing ---")
    # This part requires actual data loading, which is now in dataset.py
    # We'll simulate the inputs needed by construct_graph_nodes_and_adjacency
    print("  NOTE: This example uses simulated data. Run with train.py for actual graph construction.")

    # Simulate having loaded data
    N_SAMPLES = 500
    SEQ_LEN = 100
    N_TRAIN = 300
    simulated_signals = np.random.rand(N_SAMPLES, SEQ_LEN).astype(np.float32) * 100
    simulated_metadata = [{'global_index': i, 'label': 'Simulated'} for i in range(N_SAMPLES)]
    simulated_train_indices = list(range(N_TRAIN))

    try:
        G, node_features, updated_metadata = construct_graph_nodes_and_adjacency(
            simulated_signals,
            simulated_metadata,
            simulated_train_indices,
            n_clusters_input=CHOSEN_N_CLUSTERS, # Use placeholder K
            perform_k_analysis=False,      # Keep analysis off for simple test
            visualize_graph=True,          # Show graph plot for visual check
            
        )

        if G is not None:
            print("\n--- Graph Construction Test Summary ---")
            print(f"Graph nodes: {G.number_of_nodes()}")
            print(f"Graph edges: {G.number_of_edges()}")
            if G.number_of_nodes() > 0:
                print(f"Node features shape (centroids): {node_features.shape}")
                # Check feature assignment in graph structure
                print(f"Features shape for node 0 in graph: {G.nodes[0]['features'].shape}")
            print(f"Metadata updated with 'cluster_label': {'cluster_label' in updated_metadata[0]}")
            print("\n--- graph.py Test Finished Successfully ---")
        else:
             print("\n--- graph.py Test Failed: Graph construction returned None ---")


    except Exception as e:
        print(f"\nFATAL ERROR during graph.py test: {e}")
        traceback.print_exc()


# --- END OF FILE graph.py ---