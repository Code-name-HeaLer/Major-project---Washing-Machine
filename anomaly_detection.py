# --- START OF FILE anomaly_detection.py ---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import networkx as nx
from collections import defaultdict # Added for grouping indices
import ast # Added to parse paths string
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler # Needed if scaler was used
import traceback

# Import custom modules
import dataset
import graph # Needed for extract_features if recalculating graph stuff, but ideally load saved graph
import model

print("\n--- Anomaly Detection Script ---")

# --- Configuration (Should match training config where relevant) ---
class InferConfig:
    # Paths
    MODEL_SAVE_DIR = './models'
    BEST_MODEL_NAME = 'gcn_transformer_best.pt'
    GRAPH_SAVE_NAME = 'graph_structure.pkl'
    RESULTS_DIR = './anomaly_detection_results'
    REPORT_NAME = 'anomaly_report.txt'

    # Data Parameters
    SIGNAL_COLUMN_INDEX = 2 # Must match training
    BATCH_SIZE = 64 # Can be larger for inference
    NUM_WORKERS = 0

    # --- Data Files for Inference ---
    # <<< CHANGED: Removed UH_repeated_cycle file >>>
    paths_str_inference = """{ './dataset/11_REFIT_B2_WM_healthy_activations.npz': 'Healthy' ,
                 './dataset/11_REFIT_B2_WM_unhealthy_high_energy_activations.npz': 'UH_high_energy',
                 './dataset/11_REFIT_B2_WM_unhealthy_low_extended_energy_activations.npz' : 'UH_low_energy'

               }"""
    # Original including repeated:
    # paths_str_inference = """{ './dishwasher-dataset/11_REFIT_B2_DW_healthy_activations.npz': 'Healthy' ,
    #              './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_high_energy_activations.npz': 'UH_high_energy',
    #              './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_low_extended_energy_activations.npz' : 'UH_low_energy',
    #              './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_noisy_activations.npz' : 'UH_noisy',
    #              './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_repeated_cycle_activations.npz' : 'UH_repeated_cycle'
    #            }"""
    try:
        NPZ_FILE_PATHS_INFERENCE = ast.literal_eval(paths_str_inference)
    except:
        print("FATAL: Could not parse NPZ_FILE_PATHS_INFERENCE")
        exit()

    # --- Split Counts Used During HEALTHY-ONLY Training ---
    # <<< Define the split used to generate the model/graph being loaded >>>
    # <<< This determines which 40 healthy samples form the validation set >>>
    SPLIT_COUNTS_VALIDATION_SETUP = {
        'train': {'Healthy': 180, 'Unhealthy': 0},
        'val':   {'Healthy': 40,  'Unhealthy': 0},
        'test':  {'Healthy': 0,   'Unhealthy': 0}
    }
    HEALTHY_NPZ_PATH = './dataset/11_REFIT_B2_WM_healthy_activations.npz' # Path to the healthy file


    # Thresholding and Evaluation
    THRESHOLD_METHOD = 'F1_OPTIMAL' # Options: 'F1_OPTIMAL', 'QUANTILE', 'MAX_HEALTHY'
    QUANTILE = 0.99 # Quantile for QUANTILE method
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Plotting
    SAVE_PLOTS = True
    PLOT_FORMAT = 'png' # e.g., 'png', 'pdf', 'svg'
    N_PLOT_SAMPLES = 6 # <<< CHANGED: Number of examples per plot category


# --- Helper Functions ---

def load_model_and_graph(config):
    """Loads the trained model, graph data, and training config."""
    print("\n--- Loading Model and Graph Data ---")
    # --- PART 1: Define Paths and Check Existence (DO NOT REMOVE) ---
    model_path = os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)
    graph_path = os.path.join(config.MODEL_SAVE_DIR, config.GRAPH_SAVE_NAME)
    config_path = os.path.join(config.MODEL_SAVE_DIR, 'train_config.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph data not found at: {graph_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found at: {config_path}")
    # --- END OF PART 1 ---

    # --- PART 2: Load Config and Graph Data ---
    # Load training config (assuming it was saved with UPPERCASE keys now)
    try:
        with open(config_path, 'r') as f:
            train_config_dict = json.load(f)
        print("  Loaded training configuration.")
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Could not decode train_config.json: {e}")
        raise e
    except Exception as e:
        print(f"FATAL ERROR loading train_config.json: {e}")
        raise e


    # Load graph data
    try:
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
            graph_obj = graph_data['graph']
            node_features_centroids = graph_data['node_features']
            scaler = graph_data.get('scaler')
            # <<<--- LOAD METADATA WITH LABELS --- >>>
            # This metadata contains cluster labels based on the healthy-only training graph
            all_metadata_with_labels = graph_data.get('all_metadata_with_labels')
            if all_metadata_with_labels is None:
                raise ValueError("'all_metadata_with_labels' not found in graph_structure.pkl. "
                                 "Please ensure it was saved during training or regenerate the file.")
            print("  Loaded graph structure, node features, scaler, and metadata with labels.")
    except Exception as e:
        print(f"FATAL ERROR loading graph_structure.pkl: {e}")
        raise e
    # --- END OF PART 2 ---


    # --- PART 3: Initialize Model (This replaces the previous initialization block) ---
    print("  Initializing model architecture...")
    try:
        # Check if essential keys exist before initializing
        required_keys = ['INPUT_DIM', 'SEQ_LEN', 'NODE_FEATURE_DIM', 'N_CLUSTERS', 'D_MODEL', 'NHEAD',
                         'NUM_ENCODER_LAYERS', 'NUM_DECODER_LAYERS', 'DIM_FEEDFORWARD', 'GCN_HIDDEN_DIM',
                         'GCN_OUT_DIM', 'GCN_LAYERS', 'DROPOUT']
        missing_keys = [k for k in required_keys if k not in train_config_dict or train_config_dict[k] is None]
        if missing_keys:
            raise KeyError(f"Essential keys missing or None in loaded train_config.json: {missing_keys}")

        model_instance = model.GCNTransformerAutoencoder(
            input_dim=train_config_dict['INPUT_DIM'],
            seq_len=train_config_dict['SEQ_LEN'],
            node_feature_dim=train_config_dict['NODE_FEATURE_DIM'],
            n_clusters=train_config_dict['N_CLUSTERS'],
            d_model=train_config_dict['D_MODEL'],
            nhead=train_config_dict['NHEAD'],
            num_encoder_layers=train_config_dict['NUM_ENCODER_LAYERS'],
            num_decoder_layers=train_config_dict['NUM_DECODER_LAYERS'],
            dim_feedforward=train_config_dict['DIM_FEEDFORWARD'],
            gcn_hidden_dim=train_config_dict['GCN_HIDDEN_DIM'],
            gcn_out_dim=train_config_dict['GCN_OUT_DIM'],
            gcn_layers=train_config_dict['GCN_LAYERS'],
            dropout=train_config_dict['DROPOUT']
        )
        print("  Model architecture initialized.")
    except KeyError as ke:
        print(f"FATAL ERROR: Missing key in loaded train_config.json: {ke}")
        print("Please ensure train_config.json contains all necessary model parameters with correct (UPPERCASE) keys.")
        raise ke # Re-raise the error after printing info
    except Exception as e:
         print(f"FATAL ERROR during model instantiation: {e}")
         raise e
    # --- END OF PART 3 ---


    # --- PART 4: Load Model Weights and Prepare Tensors ---
    # Load model state dictionary
    try:
        checkpoint = torch.load(model_path, map_location=torch.device(config.DEVICE))
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.to(config.DEVICE)
        model_instance.eval() # Set model to evaluation mode
        print(f"  Loaded model weights from epoch {checkpoint['epoch']} with best val loss {checkpoint['best_val_loss']:.4f}")
    except Exception as e:
         print(f"FATAL ERROR loading model state_dict from {model_path}: {e}")
         raise e


    # Prepare graph tensors
    try:
        node_features_tensor = torch.tensor(node_features_centroids, dtype=torch.float).to(config.DEVICE)
        adj_matrix = nx.to_numpy_array(graph_obj, weight='weight') # Get adjacency matrix with weights
        adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(config.DEVICE)
        print("  Prepared graph tensors for inference.")
    except Exception as e:
         print(f"FATAL ERROR preparing graph tensors: {e}")
         raise e
    # --- END OF PART 4 ---

    print("--- Loading Complete ---")
    # Return the loaded components including the config dictionary and the metadata with labels
    return model_instance, graph_obj, node_features_centroids, adj_matrix_tensor, scaler, train_config_dict, all_metadata_with_labels


def calculate_reconstruction_errors(model, dataloader, criterion, device, node_features_tensor, adj_matrix_tensor, all_metadata_with_labels):
    """Calculates reconstruction error for each sample in the dataloader."""
    model.eval()
    errors = []
    true_labels = []
    global_indices_list = []
    local_indices_list = [] # <<< ADDED: Store local index within the dataset split

    print(f"\n--- Calculating Reconstruction Errors ({len(dataloader.dataset)} samples)---")
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (signals, labels, global_indices) in enumerate(dataloader):
            signals = signals.to(device)
            # labels = labels.to(device) # Labels needed on CPU for list storage
            # global_indices = global_indices.to(device) # Indices needed on CPU

            # Calculate local indices for this batch
            start_local_idx = batch_idx * dataloader.batch_size
            batch_local_indices = list(range(start_local_idx, start_local_idx + len(signals)))


            # Get state indices for the batch
            try:
                # Ensure we use the correct global index for metadata lookup
                state_indices = torch.tensor([all_metadata_with_labels[g_idx]['cluster_label'] for g_idx in global_indices.tolist()], dtype=torch.long).to(device)
            except IndexError:
                 print(f"Error: Global index out of bounds during error calculation (global indices: {global_indices.tolist()}, max metadata index: {len(all_metadata_with_labels)-1}). Skipping batch.")
                 continue
            except KeyError:
                 print(f"Error: 'cluster_label' not found during error calculation for indices {global_indices.tolist()}. Skipping batch.")
                 continue

            output = model(signals, state_indices, node_features_tensor, adj_matrix_tensor, tgt=signals)

            # Calculate loss per sample in the batch
            if isinstance(criterion, (nn.L1Loss, nn.MSELoss)):
                # Calculate element-wise loss, then mean over sequence and feature dims
                sample_errors = torch.mean(torch.abs(output - signals) if isinstance(criterion, nn.L1Loss) else (output - signals)**2, dim=(1, 2))
            else:
                 # Fallback: calculate batch loss and divide by batch size (less precise)
                 batch_loss = criterion(output, signals)
                 sample_errors = torch.full((signals.size(0),), batch_loss.item() / signals.size(0), device=device)


            errors.extend(sample_errors.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            global_indices_list.extend(global_indices.cpu().numpy())
            local_indices_list.extend(batch_local_indices) # <<< ADDED

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(dataloader)}")

    end_time = time.time()
    print(f"--- Error Calculation Finished (Time: {end_time - start_time:.2f}s) ---")
    # <<< CHANGED: Return local indices as well >>>
    return np.array(errors), np.array(true_labels), np.array(global_indices_list), np.array(local_indices_list)


def find_f1_optimal_threshold(errors, labels):
    """Finds the threshold that maximizes F1-score using validation errors."""
    print("\n--- Finding Optimal Threshold using F1 Score on Validation Set ---")
    fpr, tpr, thresholds = roc_curve(labels, errors)

    best_f1 = -1
    best_threshold = -1

    if len(thresholds) <= 2 or len(np.unique(labels)) < 2: # Handle cases with no separation or single class
        print("  Warning: Not enough distinct error values or only one class present in validation set to determine F1 optimal threshold reliably. Consider using QUANTILE or MAX_HEALTHY.")
        # Fallback: Use median error as a guess.
        best_threshold = np.median(errors) if len(errors)>0 else 0.0
        print(f"  Using median validation error as fallback threshold: {best_threshold:.6f}")
        return best_threshold

    # Iterate through thresholds (or a reasonable subset)
    for thresh in thresholds:
        if thresh == np.inf or thresh == -np.inf: continue # Skip infinite thresholds
        predicted = (errors >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"  Best Threshold: {best_threshold:.6f} (Max F1 Score: {best_f1:.4f})")
    print("--- Threshold Determination (F1 Optimal) Done ---")
    return best_threshold


def find_quantile_threshold(errors, labels, quantile):
    """Finds threshold based on quantile of errors from HEALTHY validation samples."""
    print(f"\n--- Finding Threshold using {quantile*100:.1f}th Quantile on Healthy Validation Set ---")
    healthy_errors = errors[labels == 0]
    if len(healthy_errors) == 0:
        raise ValueError("Cannot determine quantile threshold: No healthy samples found in validation set.")
    threshold = np.quantile(healthy_errors, quantile)
    print(f"  Threshold: {threshold:.6f}")
    print("--- Threshold Determination (Quantile) Done ---")
    return threshold

def find_max_healthy_threshold(errors, labels):
    """Finds threshold based on max error of HEALTHY validation samples."""
    print("\n--- Finding Threshold using Max Error on Healthy Validation Set ---")
    healthy_errors = errors[labels == 0]
    if len(healthy_errors) == 0:
        raise ValueError("Cannot determine max threshold: No healthy samples found in validation set.")
    threshold = np.max(healthy_errors)
    print(f"  Threshold: {threshold:.6f}")
    print("--- Threshold Determination (Max Healthy) Done ---")
    return threshold


def evaluate_performance(true_labels, predicted_labels):
    """Calculates and prints standard classification metrics."""
    print("\n--- Evaluating Performance ---")
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("--- Evaluation Done ---")
    return accuracy, precision, recall, f1, cm


# --- Plotting Functions ---

def plot_error_distribution(errors, labels, threshold, config):
    """Plots the distribution of reconstruction errors for healthy and unhealthy samples."""
    plt.figure(figsize=(10, 6))
    sns.histplot(errors[labels == 0], color="blue", label="Healthy (Normal)", kde=True, stat="density", linewidth=0)
    sns.histplot(errors[labels == 1], color="red", label="Unhealthy (Anomaly)", kde=True, stat="density", linewidth=0)
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title('Reconstruction Error Distribution (Test Set)')
    plt.xlabel('Reconstruction Error (MAE/MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':')
    if config.SAVE_PLOTS:
        plt.savefig(os.path.join(config.RESULTS_DIR, f'error_distribution.{config.PLOT_FORMAT}'))
    plt.show()


def plot_errors_over_time(errors, threshold, global_indices, config):
    """Plots reconstruction errors over sample index."""
    plt.figure(figsize=(15, 6))
    plt.plot(global_indices, errors, label='Reconstruction Error', color='blue', marker='.', linestyle='-', markersize=3, linewidth=0.5)
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    # Need test_labels to color points correctly here - assuming indices align
    # We need to map global_indices back to their original labels if needed, or pass test_labels
    # For simplicity, let's just mark those above threshold
    anomalous_idx_plot = np.where(errors > threshold)[0]
    if len(anomalous_idx_plot) > 0:
        plt.scatter(global_indices[anomalous_idx_plot], errors[anomalous_idx_plot], color='red', marker='x', s=50, label='Anomaly Detected')

    plt.title('Reconstruction Error Over Test Samples')
    plt.xlabel('Global Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.grid(True, linestyle=':')
    if config.SAVE_PLOTS:
        plt.savefig(os.path.join(config.RESULTS_DIR, f'errors_over_time.{config.PLOT_FORMAT}'))
    plt.show()


def plot_signal_reconstruction(signals_original, signals_reconstructed, local_indices_to_plot, test_global_indices_map, all_metadata, errors, threshold, config, label_type):
    """Plots original vs. reconstructed signals for selected samples.

    Args:
        signals_original: The original (unscaled) signals array (globally indexed).
        signals_reconstructed: The reconstructed signals array (locally indexed based on test set).
        local_indices_to_plot: Array of *local* indices (within the test set) to plot.
        test_global_indices_map: Array mapping local test indices to global indices.
        all_metadata: Metadata array (globally indexed).
        errors: Reconstruction errors array (locally indexed based on test set).
        threshold: Anomaly threshold.
        config: Configuration object.
        label_type: String label for the plot title/filename (e.g., 'Normal', 'HighEnergyAnomaly').
    """
    n_samples = len(local_indices_to_plot)
    if n_samples == 0:
        print(f"No {label_type} samples provided to plot.")
        return

    # --- Layout Change: Use 2 columns ---
    ncols = 2
    nrows = (n_samples + ncols - 1) // ncols # Calculate rows needed, ceiling division
    # Adjust figure size for better spacing with 2 columns
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4), sharex=True)
    # Flatten axes array for easy iteration, handle case of single row/col
    axes = axes.flatten() if n_samples > 1 else [axes]
    # --- End Layout Change ---

    # Ensure reconstruction and error arrays match the expected local indexing length
    n_test_samples = len(test_global_indices_map)
    # Validate array lengths (important!)
    if signals_reconstructed.shape[0] != n_test_samples:
        print(f"FATAL Error in plot_signal_reconstruction: signals_reconstructed length ({signals_reconstructed.shape[0]}) != test_global_indices_map length ({n_test_samples}). Cannot plot.")
        return
    if errors.shape[0] != n_test_samples:
        print(f"FATAL Error in plot_signal_reconstruction: errors length ({errors.shape[0]}) != test_global_indices_map length ({n_test_samples}). Cannot plot.")
        return

    seq_len = signals_original.shape[1]
    time_steps = np.arange(seq_len)

    plotted_count = 0
    for i, local_idx in enumerate(local_indices_to_plot):
        # Validate local_idx
        if local_idx < 0 or local_idx >= n_test_samples:
            print(f"Warning: Skipping invalid local index {local_idx} for plotting {label_type}.")
            continue

        # Get the corresponding global index
        try:
            global_idx = test_global_indices_map[local_idx]
        except IndexError:
            print(f"Warning: Could not map local index {local_idx} to global index. Skipping plot for {label_type}.")
            continue

        # Validate global_idx for original signals and metadata
        if global_idx < 0 or global_idx >= signals_original.shape[0] or global_idx >= len(all_metadata):
            print(f"Warning: Skipping invalid global index {global_idx} (mapped from local {local_idx}) for plotting {label_type}.")
            continue

        # --- Use flattened axes array index ---
        # Check if plotted_count exceeds available axes (shouldn't if nrows*ncols is correct)
        if plotted_count >= len(axes):
             print(f"Warning: Exceeded available axes ({len(axes)}) while plotting sample {plotted_count+1}/{n_samples} for {label_type}. Skipping remaining plots.")
             break
        ax = axes[plotted_count]
        # --- End Use flattened axes ---

        # Access original signal and metadata using global_idx
        original = signals_original[global_idx].flatten()
        metadata_entry = all_metadata[global_idx]
        true_detailed_label = metadata_entry.get('label', 'N/A') # Use .get for safety
        key = metadata_entry.get('source_key', 'N/A')

        # Access reconstructed signal and error using local_idx
        reconstructed = signals_reconstructed[local_idx].flatten()
        error = errors[local_idx]
        anomaly_detected = error > threshold

        ax.plot(time_steps, original, label='Original Signal', color='blue', linewidth=1.0) # Slightly thinner lines might help
        ax.plot(time_steps, reconstructed, label=f'Reconstructed (Err: {error:.4f})', color='red', linestyle='--', linewidth=1.0)
        ax.set_title(f'Window {global_idx} (Local Test Idx: {local_idx}) - True Label: {true_detailed_label} (Key: {key}) - Detected: {"Anomaly" if anomaly_detected else "Normal"}')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.set_ylabel('Signal Value')
        plotted_count += 1

    # Adjust layout and save only if plots were actually generated
    if plotted_count > 0:
        # Remove unused subplots if n_samples is odd or some were skipped
        for j in range(plotted_count, nrows * ncols):
             try:
                  fig.delaxes(axes[j])
             except IndexError: # Avoid error if axes array was smaller than expected
                  print(f"Warning: Could not delete axis {j}, index out of bounds.")


        # Set xlabel only on the bottom-most plots that were actually used
        start_xlabel_row = max(0, plotted_count - ncols)
        for ax_idx in range(start_xlabel_row, plotted_count):
            try:
                axes[ax_idx].set_xlabel('Time Step')
            except IndexError:
                 print(f"Warning: Could not set xlabel for axis {ax_idx}, index out of bounds.")

        fig.suptitle(f'Original vs. Reconstructed Signals ({label_type} Examples)', fontsize=16)
        # Use tight_layout BEFORE potentially saving/showing to adjust spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if config.SAVE_PLOTS:
            # <<< CHANGED: Use label_type in filename >>>
            save_path = os.path.join(config.RESULTS_DIR, f'reconstruction_{label_type}.{config.PLOT_FORMAT}')
            try:
                 plt.savefig(save_path)
                 print(f"Saved reconstruction plot to: {save_path}")
            except Exception as e:
                 print(f"Error saving reconstruction plot to {save_path}: {e}")
        plt.show()
    elif n_samples > 0: # No plots generated, but axes might exist
         plt.close(fig) # Close the empty figure


def plot_confusion_matrix(cm, class_names, config):
    """Plots the confusion matrix."""
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if config.SAVE_PLOTS:
        plt.savefig(os.path.join(config.RESULTS_DIR, f'confusion_matrix.{config.PLOT_FORMAT}'))
    plt.show()

def plot_performance_metrics(accuracy, precision, recall, f1, config):
    """Plots performance metrics in a bar chart."""
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    plt.ylabel('Score')
    plt.title('Anomaly Detection Performance Metrics (Test Set)')
    plt.ylim([0, 1.1])
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
    if config.SAVE_PLOTS:
        plt.savefig(os.path.join(config.RESULTS_DIR, f'performance_metrics.{config.PLOT_FORMAT}'))
    plt.show()

# --- Main Inference Function ---
def main(config):
    """Main function for anomaly detection."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, config.REPORT_NAME)

    start_total_time = time.time()
    report_lines = ["Anomaly Detection Report\n", "="*30 + "\n"]

    # 1. Load Model, Graph, Config, Scaler (Ensure these are from healthy-only training)
    try:
        model_instance, graph_obj, node_features_centroids, adj_matrix_tensor, scaler, train_config_dict, all_metadata_with_labels_original = load_model_and_graph(config)
        report_lines.append(f"Loaded model: {config.BEST_MODEL_NAME} (Assumed trained on healthy only)\n")
        report_lines.append(f"Loaded graph data: {config.GRAPH_SAVE_NAME} (Assumed built on healthy only)\n")
        report_lines.append(f"Using device: {config.DEVICE}\n")
        if node_features_centroids is not None:
            actual_n_clusters = node_features_centroids.shape[0]
            print(f"Inferred N_CLUSTERS from loaded centroids: {actual_n_clusters}")
            node_features_tensor = torch.tensor(node_features_centroids, dtype=torch.float).to(config.DEVICE)
        else:
             actual_n_clusters = train_config_dict.get('N_CLUSTERS', None)
             print(f"Could not infer N_CLUSTERS from centroids, using value from train_config: {actual_n_clusters}")
             node_features_tensor = None
        if actual_n_clusters is None: raise ValueError("Could not determine N_CLUSTERS")
        if node_features_tensor is None and actual_n_clusters > 0:
             raise ValueError("Node features centroids were missing or invalid, cannot create node_features_tensor.")
        expected_seq_len = train_config_dict.get('SEQ_LEN')
        if expected_seq_len is None:
            raise ValueError("Could not determine expected SEQ_LEN from loaded training config.")
        print(f"Model expects sequence length: {expected_seq_len}")
    except Exception as e:
        print(f"FATAL ERROR loading model/graph: {e}"); traceback.print_exc(); return

    # 2. Load Data for Inference (NOW EXCLUDES REPEATED CYCLE)
    try:
        print("\n--- Loading Data for Inference (Excluding Repeated Cycle) ---")
        signals_padded_raw_original_len, all_metadata_raw, max_len_in_files = dataset.load_and_pad_signals(
            config.NPZ_FILE_PATHS_INFERENCE, config.SIGNAL_COLUMN_INDEX, dataset.PADDING_VALUE
        )
        print(f"Total samples loaded for inference: {len(all_metadata_raw)}. Max length found in files: {max_len_in_files}")

        # Adjust sequence length
        print(f"Adjusting loaded sequences to expected length: {expected_seq_len}...")
        current_len = signals_padded_raw_original_len.shape[1]
        if current_len > expected_seq_len:
            print(f"  Truncating sequences from {current_len} to {expected_seq_len}.")
            signals_padded_raw = signals_padded_raw_original_len[:, :expected_seq_len]
        elif current_len < expected_seq_len:
            print(f"  Padding sequences from {current_len} to {expected_seq_len}.")
            pad_width = expected_seq_len - current_len
            signals_padded_raw = np.pad(signals_padded_raw_original_len, ((0, 0), (0, pad_width)), 'constant', constant_values=dataset.PADDING_VALUE)
        else:
            print(f"  Loaded sequence length ({current_len}) matches expected length ({expected_seq_len}). No adjustment needed.")
            signals_padded_raw = signals_padded_raw_original_len
        print(f"  Final signal shape for processing: {signals_padded_raw.shape}")

        # Apply signal scaling
        if scaler:
            print("Applying loaded SIGNAL scaler (fitted on healthy data)...")
            signals_padded_scaled = dataset.scale_data(signals_padded_raw, scaler, dataset.PADDING_VALUE)
            scale_status = "Applied (from healthy train)"
        else:
            print("No signal scaler loaded/applied.")
            signals_padded_scaled = signals_padded_raw
            scale_status = "Not Applied"
        report_lines.append(f"Signal Scaling: {scale_status}\n")

        # Determine Validation and Test Indices
        print("\n--- Determining Validation and Test Indices ---")
        healthy_metadata_raw_for_split = [m for m in all_metadata_raw if m['label'] == 'Healthy']
        if len(healthy_metadata_raw_for_split) < 400:
            print(f"Warning: Expected >= 400 healthy samples, but found {len(healthy_metadata_raw_for_split)} in loaded data.")
        _, val_indices_healthy_relative, _ = dataset.split_data_indices(
            healthy_metadata_raw_for_split, config.SPLIT_COUNTS_VALIDATION_SETUP, dataset.RANDOM_SEED
        )
        healthy_split_to_global_map = {idx: meta['global_index'] for idx, meta in enumerate(healthy_metadata_raw_for_split)}
        val_indices = [healthy_split_to_global_map[i] for i in val_indices_healthy_relative if i in healthy_split_to_global_map]
        print(f"Validation Set: Using {len(val_indices)} indices identified from healthy-only split.")
        if len(val_indices) != 40: print(f"Warning: Expected 40 validation indices, got {len(val_indices)}")

        # Construct Test Indices (ALL Healthy + REMAINING Unhealthy)
        test_indices_healthy = [m['global_index'] for m in all_metadata_raw if m['label'] == 'Healthy']
        test_indices_unhealthy = [m['global_index'] for m in all_metadata_raw if m['label'] != 'Healthy']
        test_indices = sorted(test_indices_healthy + test_indices_unhealthy)
        print(f"Test Set: Using {len(test_indices_healthy)} Healthy + {len(test_indices_unhealthy)} Unhealthy (excluding repeated) = {len(test_indices)} total indices.") # Updated print

        # Generate Cluster Labels for ALL loaded Inference Samples
        print("\n--- Generating Cluster Labels for All Inference Samples via Nearest Centroid ---")
        print("  Extracting features for all inference samples...")
        all_features_raw = np.array([graph.extract_features(window) for window in signals_padded_raw])
        if all_features_raw.shape[1] != 15: raise ValueError(f"Expected 15 features, but got {all_features_raw.shape[1]}")
        print("  Fitting FEATURE scaler on healthy validation samples...")
        valid_val_indices = [idx for idx in val_indices if idx < all_features_raw.shape[0]]
        if not valid_val_indices: raise ValueError("No valid healthy validation sample indices found to fit feature scaler.")
        val_features_raw = all_features_raw[valid_val_indices]
        if val_features_raw.shape[0] == 0: raise ValueError("No healthy validation features extracted to fit feature scaler.")
        feature_scaler_inference = StandardScaler()
        feature_scaler_inference.fit(val_features_raw)
        print(f"  Feature scaler fitted on {len(valid_val_indices)} samples.")
        print("  Scaling all inference features...")
        all_features_scaled = feature_scaler_inference.transform(all_features_raw)
        all_features_scaled = np.nan_to_num(all_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        n_centroids = node_features_centroids.shape[0]
        print(f"  Using {n_centroids} centroids from healthy graph (assumed scaled).")
        print("  Assigning cluster labels based on nearest centroid...")
        assigned_cluster_labels = np.zeros(len(all_features_scaled), dtype=int)
        for i in range(len(all_features_scaled)):
            sample_features = all_features_scaled[i]
            distances = np.linalg.norm(node_features_centroids - sample_features.reshape(1, -1), axis=1)
            assigned_cluster_labels[i] = np.argmin(distances)
        all_metadata_with_labels_inference = []
        for i, meta_raw in enumerate(all_metadata_raw):
            new_meta = meta_raw.copy()
            new_meta['cluster_label'] = assigned_cluster_labels[i]
            all_metadata_with_labels_inference.append(new_meta)
        print(f"  Generated metadata with assigned cluster labels for {len(all_metadata_with_labels_inference)} samples.")

        # Create Datasets/DataLoaders
        metadata_to_use = all_metadata_with_labels_inference
        inference_label_map = {'Healthy': 0}
        for label in config.NPZ_FILE_PATHS_INFERENCE.values(): # Uses updated paths
            if label != 'Healthy': inference_label_map[label] = 1
        val_dataset = dataset.ApplianceDataset(signals_padded_scaled, metadata_to_use, val_indices, inference_label_map)
        test_dataset = dataset.ApplianceDataset(signals_padded_scaled, metadata_to_use, test_indices, inference_label_map)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        print(f"Validation DataLoader: {len(val_loader)} batches ({len(val_dataset)} samples)")
        print(f"Test DataLoader: {len(test_loader)} batches ({len(test_dataset)} samples)")

    except Exception as e:
        print(f"FATAL ERROR during data loading/prep for inference: {e}"); traceback.print_exc(); return

    # 3. Calculate Reconstruction Errors
    criterion = nn.L1Loss(reduction='none')
    val_errors, val_labels, _, val_local_indices = calculate_reconstruction_errors(model_instance, val_loader, criterion, config.DEVICE, node_features_tensor, adj_matrix_tensor, metadata_to_use)
    test_errors, test_labels, test_global_indices, test_local_indices = calculate_reconstruction_errors(model_instance, test_loader, criterion, config.DEVICE, node_features_tensor, adj_matrix_tensor, metadata_to_use)

    # 4. Determine Threshold (using the 40 healthy validation samples)
    unique_val_labels = np.unique(val_labels)
    if not (len(unique_val_labels) == 1 and unique_val_labels[0] == 0):
        print(f"Warning: Validation set used for thresholding contains non-healthy labels: {unique_val_labels}. Threshold might be unreliable.")
    threshold_method_used = config.THRESHOLD_METHOD
    if config.THRESHOLD_METHOD == 'F1_OPTIMAL':
        print("\nWarning: F1_OPTIMAL is not suitable for healthy-only validation set. Using QUANTILE instead.")
        threshold_method_used = 'QUANTILE'
        try: threshold = find_quantile_threshold(val_errors, val_labels, config.QUANTILE)
        except ValueError as e: print(f"Error finding quantile threshold: {e}. Falling back to median."); threshold = np.median(val_errors) if len(val_errors) > 0 else 0.0
    elif config.THRESHOLD_METHOD == 'QUANTILE':
        try: threshold = find_quantile_threshold(val_errors, val_labels, config.QUANTILE)
        except ValueError as e: print(f"Error finding quantile threshold: {e}. Falling back to median."); threshold = np.median(val_errors) if len(val_errors) > 0 else 0.0
    elif config.THRESHOLD_METHOD == 'MAX_HEALTHY':
        try: threshold = find_max_healthy_threshold(val_errors, val_labels)
        except ValueError as e: print(f"Error finding max healthy threshold: {e}. Falling back to median."); threshold = np.median(val_errors) if len(val_errors) > 0 else 0.0
    else: raise ValueError(f"Unknown threshold method: {config.THRESHOLD_METHOD}")
    report_lines.append(f"Threshold Method Used: {threshold_method_used} (Based on {len(val_dataset)} healthy validation samples)\n")
    report_lines.append(f"Anomaly Threshold: {threshold:.6f}\n")


    # 5. Classify Test Samples (now excluding repeated cycle)
    test_predictions = (test_errors >= threshold).astype(int)
    report_lines.append(f"Test Set Size: {len(test_labels)}\n")
    report_lines.append(f"Predicted Anomalies in Test Set: {np.sum(test_predictions)}/{len(test_predictions)}\n")

    # 6. Evaluate Performance (on the reduced test set)
    accuracy, precision, recall, f1, cm = evaluate_performance(test_labels, test_predictions)
    report_lines.append("\n--- Performance Metrics ---\n")
    report_lines.append(f"Accuracy:  {accuracy:.4f}\n")
    report_lines.append(f"Precision: {precision:.4f}\n")
    report_lines.append(f"Recall:    {recall:.4f}\n")
    report_lines.append(f"F1 Score:  {f1:.4f}\n")
    report_lines.append(f"Confusion Matrix:\n{cm}\n")


    # 7. Generate Plots (using the reduced test set results)
    print("\n--- Generating Plots ---")
    # Plot 1: Error distribution
    plot_error_distribution(test_errors, test_labels, threshold, config)

    # Plot 2: Errors over time/index
    plot_errors_over_time(test_errors, threshold, test_global_indices, config)

    # --- Plot 3: Signal Reconstructions ---
    signals_plot_original = signals_padded_raw
    print("Generating reconstructed signals for plotting...")
    signals_reconstructed_list = []
    with torch.no_grad():
        for signals_scaled, _, global_indices in test_loader:
            signals_scaled = signals_scaled.to(config.DEVICE)
            try:
                state_indices = torch.tensor([metadata_to_use[g_idx]['cluster_label'] for g_idx in global_indices.tolist()], dtype=torch.long).to(config.DEVICE)
            except Exception as e:
                print(f"Error getting state indices during plot reconstruction: {e}. Skipping batch."); continue
            output_scaled = model_instance(signals_scaled, state_indices, node_features_tensor, adj_matrix_tensor, tgt=signals_scaled)
            signals_reconstructed_list.append(output_scaled.cpu().numpy())

    if not signals_reconstructed_list:
        print("WARNING: No reconstructed signals generated for plotting. Skipping reconstruction plots.")
    else:
        signals_reconstructed_scaled = np.concatenate(signals_reconstructed_list, axis=0)
        if signals_reconstructed_scaled.shape[0] != len(test_dataset):
             print(f"Warning: Reconstructed signals for plot ({signals_reconstructed_scaled.shape[0]}) doesn't match test dataset size ({len(test_dataset)}).")
             min_len = min(signals_reconstructed_scaled.shape[0], len(test_dataset))
             signals_reconstructed_scaled = signals_reconstructed_scaled[:min_len]
             test_errors = test_errors[:min_len]
             test_global_indices = test_global_indices[:min_len]
             test_local_indices = test_local_indices[:min_len]

        # Inverse transform
        signals_plot_reconstructed = np.copy(signals_reconstructed_scaled)
        if scaler:
            print("Applying inverse scaling to reconstructed signals for plotting...")
            num_skipped_inverse = 0
            for i in range(len(signals_plot_reconstructed)):
                signal = signals_plot_reconstructed[i].flatten()
                global_idx = test_global_indices[i]
                if global_idx >= len(signals_padded_raw): print(f"Warning: Global index {global_idx} out of bounds for raw signals. Skipping inverse scaling."); num_skipped_inverse += 1; continue
                try:
                    original_signal_raw = signals_padded_raw[global_idx].flatten()
                    non_padded_mask = (original_signal_raw != dataset.PADDING_VALUE)
                    if np.any(non_padded_mask):
                        data_to_transform = signal[non_padded_mask].reshape(-1, 1)
                        if data_to_transform.shape[1] == scaler.n_features_in_: signal[non_padded_mask] = scaler.inverse_transform(data_to_transform).flatten()
                        else: print(f"Warning: Shape mismatch for inverse transform at index {i}. Skipping."); num_skipped_inverse += 1; continue
                    signals_plot_reconstructed[i] = signal.reshape(-1, 1)
                except Exception as e: print(f"Error during inverse scaling for sample {i}: {e}"); num_skipped_inverse += 1; continue
            if num_skipped_inverse > 0: print(f"Warning: Skipped inverse scaling for {num_skipped_inverse} samples.")

        # --- Generate Specific Reconstruction Plots (Excluding RepeatedCycle) ---
        print("\n--- Generating Specific Reconstruction Plots ---")
        metadata_for_plot = all_metadata_raw
        local_to_meta = {local_idx: metadata_for_plot[global_idx]
                         for local_idx, global_idx in zip(test_local_indices, test_global_indices)
                         if global_idx < len(metadata_for_plot)}
        indices_by_label = defaultdict(list)
        for local_idx, meta in local_to_meta.items():
            indices_by_label[meta.get('label', 'Unknown')].append(local_idx)

        # <<< CHANGED: Removed UH_repeated_cycle >>>
        plot_categories = {
            'Healthy': 'Normal',
            'UH_high_energy': 'HighEnergyAnomaly',
            'UH_noisy': 'NoisyAnomaly',
            # 'UH_repeated_cycle': 'RepeatedCycleAnomaly',
            'UH_low_energy': 'LowEnergyExtendedAnomaly'
        }
        for label_key in plot_categories:
            if label_key not in indices_by_label:
                 print(f"Note: No test samples found with label '{label_key}'.")

        # Plot each remaining category
        for detailed_label, plot_suffix in plot_categories.items():
            if detailed_label in indices_by_label:
                local_indices_category = indices_by_label[detailed_label]
                n_available = len(local_indices_category)
                if n_available > 0:
                    n_to_plot = min(config.N_PLOT_SAMPLES, n_available)
                    plot_indices_local = np.random.choice(local_indices_category, n_to_plot, replace=False)
                    print(f"Plotting {n_to_plot}/{n_available} '{detailed_label}' examples ({plot_suffix})...")
                    valid_plot_indices = [idx for idx in plot_indices_local if idx < len(signals_plot_reconstructed) and idx < len(test_errors)]
                    if len(valid_plot_indices) != n_to_plot: print("Warning: Some indices out of bounds for plotting.")
                    if valid_plot_indices:
                         plot_signal_reconstruction(
                             signals_plot_original, signals_plot_reconstructed,
                             valid_plot_indices, test_global_indices, metadata_for_plot,
                             test_errors, threshold, config, plot_suffix
                         )
                    else:
                         print(f"No valid indices to plot for {plot_suffix}.")
                else:
                     print(f"No samples for '{detailed_label}'.")


    # Plot 4: Confusion Matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(cm, class_names=['Normal (0)', 'Anomaly (1)'], config=config)

    # Plot 5: Performance Metrics Bar Chart
    print("Plotting performance metrics...")
    plot_performance_metrics(accuracy, precision, recall, f1, config=config)

    # --- 8. Save Report ---
    print(f"\n--- Saving Report to {report_path} ---")
    total_time = time.time() - start_total_time
    report_lines.append(f"\nTotal Anomaly Detection Time: {total_time:.2f} seconds\n")
    try:
        with open(report_path, 'w') as f: f.writelines(report_lines)
        print("Report saved.")
    except Exception as e: print(f"Error saving report: {e}")

    print("\n--- Anomaly Detection Script Completed ---")

# --- Run Inference ---
if __name__ == "__main__":
    config = InferConfig()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    try:
        main(config)
    except FileNotFoundError as fnf_error:
        print(f"\nFATAL ERROR: File Not Found. {fnf_error}")
        print(f"Ensure model ('{config.BEST_MODEL_NAME}'), graph ('{config.GRAPH_SAVE_NAME}'), config ('train_config.json') exist in '{config.MODEL_SAVE_DIR}'.")
        print(f"Ensure NPZ files defined in InferConfig.NPZ_FILE_PATHS_INFERENCE exist.")
    except Exception as e:
        print(f"\nFATAL ERROR during anomaly detection execution: {e}")
        traceback.print_exc()

# --- END OF FILE anomaly_detection.py ---