# --- START OF FILE train.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
import pickle
import traceback
import networkx as nx
from sklearn.preprocessing import StandardScaler

# Import custom modules
import dataset
import graph
import model

print("\n--- GCN-Transformer Training Script ---")

# --- Configuration ---
class TrainConfig:
    # Data Parameters (mostly from dataset.py/graph.py, ensure consistency)
    SIGNAL_COLUMN_INDEX = 2
    BATCH_SIZE = 32 # From readme
    SCALE_DATA = True # Apply StandardScaler to signals

    # Graph Parameters (mostly from graph.py, ensure consistency)
    CHOSEN_N_CLUSTERS = 9 # Placeholder - ideally load from graph analysis/config
    NODE_FEATURE_DIM = 15 # Fixed by extract_features in graph.py
    VISUALIZE_GRAPH = False # Set to True to see graph structure plot during setup
    VISUALIZE_ASSIGNMENTS = False # Set to True to see window assignment plots

    # Model Parameters (matching readme and model.py)
    INPUT_DIM = 1 # Univariate time series
    # SEQ_LEN will be determined from data
    D_MODEL = 128 # Transformer embedding dimension (example, tune)
    NHEAD = 8 # Number of attention heads (example, must divide D_MODEL)
    NUM_ENCODER_LAYERS = 2 # Transformer encoder layers (from readme)
    NUM_DECODER_LAYERS = 2 # Transformer decoder layers (from readme)
    DIM_FEEDFORWARD = 256 # Transformer feedforward dim (example, tune)
    GCN_HIDDEN_DIM = 64 # GCN hidden dimension (example, tune)
    GCN_OUT_DIM = 32 # GCN output embedding dimension (example, tune)
    GCN_LAYERS = 8 # From readme
    DROPOUT = 0.1

    # Training Parameters
    LEARNING_RATE = 0.001 # From readme (Adam default is often 0.001)
    EPOCHS = 100 # Maximum number of epochs
    PATIENCE = 10 # Early stopping patience (from readme)
    OPTIMIZER = 'Adam' # From readme
    LOSS_FN = 'MAE' # From readme (Mean Absolute Error)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0 # DataLoader workers (set > 0 if I/O is bottleneck)

    # Paths
    MODEL_SAVE_DIR = './models'
    BEST_MODEL_NAME = 'gcn_transformer_best.pt'
    FINAL_MODEL_NAME = 'gcn_transformer_final.pt'
    GRAPH_SAVE_NAME = 'graph_structure.pkl' # For graph object, node features, scaler

# --- Helper Functions ---
def save_checkpoint(state, is_best, save_dir, best_filename, final_filename):
    """Saves model and training parameters."""
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, final_filename)
    torch.save(state, final_path)
    if is_best:
        best_path = os.path.join(save_dir, best_filename)
        torch.save(state, best_path)
        print(f"  => Saved new best model to {best_path}")
    else:
         print(f"  => Saved checkpoint to {final_path}")


def save_graph_data(graph_obj, node_features, scaler, all_metadata_with_labels, save_dir, filename): # Add all_metadata_with_labels
    """Saves graph structure, node features, scaler, and metadata with labels."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    data_to_save = {
        'graph': graph_obj,
        'node_features': node_features,
        'scaler': scaler, # Save scaler used for signals (if any)
        'all_metadata_with_labels': all_metadata_with_labels # <<< ADD THIS LINE
    }
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"  => Saved graph data (including metadata with labels) to {filepath}")
    except Exception as e:
         print(f"Error saving graph data to {filepath}: {e}")


def train_epoch(model, dataloader, optimizer, criterion, device, node_features_tensor, adj_matrix_tensor, all_metadata_with_labels):
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (signals, _, global_indices) in enumerate(dataloader):
        signals = signals.to(device) # Target is also signals for reconstruction
        global_indices = global_indices.to(device) # Keep indices on CPU/GPU as needed

        # --- Get state indices for the batch ---
        # This needs to map global_indices back to their cluster labels efficiently
        # Assuming all_metadata_with_labels is accessible
        try:
             # state_indices = torch.tensor([all_metadata_with_labels[g_idx.item()]['cluster_label'] for g_idx in global_indices], dtype=torch.long).to(device)
             # More efficient lookup if metadata is indexed by global_index or we pre-create a mapping
             # Let's assume metadata list is ordered by global index 0 to N-1 for simplicity here
             # (If not, create a lookup dict: global_idx_to_cluster = {m['global_index']: m['cluster_label'] for m in all_metadata_with_labels})
             state_indices = torch.tensor([all_metadata_with_labels[g_idx]['cluster_label'] for g_idx in global_indices.tolist()], dtype=torch.long).to(device)

        except IndexError:
             print(f"Error: Global index out of bounds when fetching cluster label.")
             print(f"Max global index: {len(all_metadata_with_labels)-1}")
             print(f"Problematic indices in batch: {global_indices}")
             # Handle error appropriately - skip batch or raise
             continue
        except KeyError:
             print(f"Error: 'cluster_label' not found in metadata for an index.")
             # Handle error
             continue


        optimizer.zero_grad()
        # Forward pass: tgt=signals for autoencoder reconstruction
        output = model(signals, state_indices, node_features_tensor, adj_matrix_tensor, tgt=signals)

        loss = criterion(output, signals)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 5 == 0: # Print progress every 5 batches
            elapsed = time.time() - start_time
            print(f'  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
            start_time = time.time() # Reset timer

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_epoch(model, dataloader, criterion, device, node_features_tensor, adj_matrix_tensor, all_metadata_with_labels):
    """Runs one evaluation epoch."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for signals, _, global_indices in dataloader:
            signals = signals.to(device)
            global_indices = global_indices.to(device) # Keep indices on CPU/GPU as needed

            # Get state indices for the batch
            try:
                # state_indices = torch.tensor([all_metadata_with_labels[g_idx.item()]['cluster_label'] for g_idx in global_indices], dtype=torch.long).to(device)
                state_indices = torch.tensor([all_metadata_with_labels[g_idx]['cluster_label'] for g_idx in global_indices.tolist()], dtype=torch.long).to(device)
            except IndexError:
                 print(f"Error: Global index out of bounds during validation.")
                 continue
            except KeyError:
                 print(f"Error: 'cluster_label' not found during validation.")
                 continue


            output = model(signals, state_indices, node_features_tensor, adj_matrix_tensor, tgt=signals)
            loss = criterion(output, signals)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# --- Main Training Function ---
def main(config):
    """Main function to run data loading, graph construction, and training."""
    print(f"Using device: {config.DEVICE}")
    print("Starting main function...")

    # --- Create config dictionary for saving ---
    # Create it here inside main after config object is ready
    config_to_save = {
        'SIGNAL_COLUMN_INDEX': config.SIGNAL_COLUMN_INDEX,
        'BATCH_SIZE': config.BATCH_SIZE,
        'SCALE_DATA': config.SCALE_DATA,
        'CHOSEN_N_CLUSTERS': config.CHOSEN_N_CLUSTERS,
        'NODE_FEATURE_DIM': config.NODE_FEATURE_DIM,
        'VISUALIZE_GRAPH': config.VISUALIZE_GRAPH,
        'VISUALIZE_ASSIGNMENTS': config.VISUALIZE_ASSIGNMENTS,
        'INPUT_DIM': config.INPUT_DIM,
        # Initialize SEQ_LEN and N_CLUSTERS as None, will be updated
        'SEQ_LEN': None,
        'N_CLUSTERS': None,
        'D_MODEL': config.D_MODEL,
        'NHEAD': config.NHEAD,
        'NUM_ENCODER_LAYERS': config.NUM_ENCODER_LAYERS,
        'NUM_DECODER_LAYERS': config.NUM_DECODER_LAYERS,
        'DIM_FEEDFORWARD': config.DIM_FEEDFORWARD,
        'GCN_HIDDEN_DIM': config.GCN_HIDDEN_DIM,
        'GCN_OUT_DIM': config.GCN_OUT_DIM,
        'GCN_LAYERS': config.GCN_LAYERS,
        'DROPOUT': config.DROPOUT,
        'LEARNING_RATE': config.LEARNING_RATE,
        'EPOCHS': config.EPOCHS,
        'PATIENCE': config.PATIENCE,
        'OPTIMIZER': config.OPTIMIZER,
        'LOSS_FN': config.LOSS_FN,
        'DEVICE': config.DEVICE,
        'NUM_WORKERS': config.NUM_WORKERS,
    }
    config_path = os.path.join(config.MODEL_SAVE_DIR, 'train_config.json') # Define path here
    print(f"Config save path defined: {config_path}") # <<< DEBUG >>>

    # --- 1. Load Data ---
    print("\n--- Loading and Preprocessing Data ---")
    try:
        (train_loader, val_loader, test_loader,
         all_metadata, train_indices, val_indices, test_indices,
         max_len, scaler) = dataset.get_dataloaders(
             batch_size=config.BATCH_SIZE,
             signal_col_idx=config.SIGNAL_COLUMN_INDEX,
             scale=config.SCALE_DATA,
             num_workers=config.NUM_WORKERS
         )
        config.SEQ_LEN = max_len # Update config with actual sequence length
        config_to_save['SEQ_LEN'] = max_len
        print(f"Data loaded. Sequence length: {config.SEQ_LEN}")
    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        traceback.print_exc()
        return

    # Need padded signals for graph construction feature extraction
    # We can get this by reloading or modifying get_dataloaders to return it
    # Let's reload for simplicity now, but optimize later if memory is an issue
    print("\n--- Reloading Padded Signals for Graph Construction ---")
    try:
        signals_padded, _, _ = dataset.load_and_pad_signals(
            dataset.NPZ_FILE_PATHS, config.SIGNAL_COLUMN_INDEX, dataset.PADDING_VALUE
        )
        # Apply scaling if it was done for the dataloaders
        if config.SCALE_DATA and scaler:
             print("Applying scaler to padded signals for graph construction...")
             signals_for_graph = dataset.scale_data(signals_padded, scaler, dataset.PADDING_VALUE)
        else:
             signals_for_graph = signals_padded

    except Exception as e:
        print(f"FATAL ERROR reloading/scaling signals for graph: {e}")
        traceback.print_exc()
        return

    # --- 2. Construct Graph ---
    print("\n--- Constructing Graph ---")
    try:
        # Note: construct_graph needs all_metadata, not just train_metadata
        G, node_features_centroids, all_metadata_with_labels = graph.construct_graph_nodes_and_adjacency(
            signals_for_graph, # Use potentially scaled signals
            all_metadata.copy(), # Pass a copy to avoid modifying original
            train_indices,
            n_clusters_input=config.CHOSEN_N_CLUSTERS,
            perform_k_analysis=False, # Keep analysis off unless specifically needed
            visualize_graph=config.VISUALIZE_GRAPH,
            visualize_assignments=config.VISUALIZE_ASSIGNMENTS
        )
        if G is None:
            raise ValueError("Graph construction failed.")

        # Update config with actual number of clusters found
        config.N_CLUSTERS = G.number_of_nodes()
        config_to_save['N_CLUSTERS'] = config.N_CLUSTERS
        print(f"Graph constructed. Number of nodes (clusters): {config.N_CLUSTERS}")

        # Prepare graph components for model input
        node_features_tensor = torch.tensor(node_features_centroids, dtype=torch.float).to(config.DEVICE)
        adj_matrix = nx.to_numpy_array(G, weight='weight') # Get adjacency matrix with weights
        adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(config.DEVICE)
        print("Graph tensors prepared.") # <<< DEBUG >>>

    except Exception as e:
        print(f"FATAL ERROR during graph construction: {e}")
        traceback.print_exc()
        return

    # --- 3. Initialize Model ---
    print("\n--- Initializing Model ---")
    try:
        model_instance = model.GCNTransformerAutoencoder(
            input_dim=config.INPUT_DIM,
            seq_len=config.SEQ_LEN,
            node_feature_dim=config.NODE_FEATURE_DIM,
            n_clusters=config.N_CLUSTERS,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            gcn_hidden_dim=config.GCN_HIDDEN_DIM,
            gcn_out_dim=config.GCN_OUT_DIM,
            gcn_layers=config.GCN_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        print(model_instance)
        num_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        print(f"Model initialized. Trainable parameters: {num_params:,}")

    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        traceback.print_exc()
        return

    # --- 4. Define Loss and Optimizer ---
    if config.LOSS_FN == 'MAE':
        criterion = nn.L1Loss() # MAE
    elif config.LOSS_FN == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.LOSS_FN}")

    if config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model_instance.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == 'AdamW':
         optimizer = optim.AdamW(model_instance.parameters(), lr=config.LEARNING_RATE)
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")

    print(f"Using Loss: {config.LOSS_FN}, Optimizer: {config.OPTIMIZER}, LR: {config.LEARNING_RATE}")
    print("Loss and optimizer defined.")

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model_instance, train_loader, optimizer, criterion, config.DEVICE,
                                 node_features_tensor, adj_matrix_tensor, all_metadata_with_labels)

        # Validate
        val_loss = evaluate_epoch(model_instance, val_loader, criterion, config.DEVICE,
                                  node_features_tensor, adj_matrix_tensor, all_metadata_with_labels)

        epoch_duration = time.time() - epoch_start_time
        print("-" * 89)
        print(f"| End of Epoch {epoch:3d} | Time: {epoch_duration:5.2f}s | "
              f"Train Loss: {train_loss:7.4f} | Val Loss: {val_loss:7.4f} |")
        print("-" * 89)

        # --- Checkpointing and Early Stopping ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_instance.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config.__dict__ # Save config with the model
            }, True, config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME, config.FINAL_MODEL_NAME)
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{config.PATIENCE}")
            # Save latest checkpoint anyway
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_instance.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss, # Save the best loss seen so far
                'config': config.__dict__
            }, False, config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME, config.FINAL_MODEL_NAME)


        if epochs_no_improve >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    training_duration = time.time() - training_start_time
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {training_duration/60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # --- 6. Save Graph Data ---
    print("\n--- Saving Graph Structure, Scaler, and Metadata with Labels ---")
    save_graph_data(G, node_features_centroids, scaler, all_metadata_with_labels, config.MODEL_SAVE_DIR, config.GRAPH_SAVE_NAME)
    print("Graph data saving attempted.") # <<< DEBUG >>>

    print("\n--- Training Script Completed ---")

    # --- 7. Save FINAL Config ---
    print(f"\n--- Saving Final Configuration to {config_path} ---")
    print(f"Dictionary to save: {config_to_save}") # <<< DEBUG >>> Check contents
    # config_to_save should now have updated SEQ_LEN and N_CLUSTERS
    try:
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print("Final configuration saved.")
    except Exception as e:
        print(f"Error saving config to JSON at the end of main: {e}")


    print("\n--- Training Script main function Completed ---")


# --- Run Training ---
if __name__ == "__main__":
    print("Script entry point reached.") # <<< DEBUG >>>
    # 1. Initialize the configuration object
    config = TrainConfig()

    # 2. Create the save directory if it doesn't exist
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    print(f"Model save directory ensured: {config.MODEL_SAVE_DIR}") # <<< DEBUG >>>

    # 3. Run the main training process
    #    The config dictionary creation, updating, and saving to JSON
    #    now happens *inside* the main function.
    try:
        print("Calling main function...") # <<< DEBUG >>>
        main(config)
        print("Returned from main function.") # <<< DEBUG >>>
    except Exception as e:
        print(f"\nFATAL ERROR during main training execution: {e}")
        traceback.print_exc()

    print("Script execution finished.") # <<< DEBUG >>> Add one final message

# --- END OF FILE train.py ---