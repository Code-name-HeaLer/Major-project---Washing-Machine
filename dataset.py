# --- START OF FILE dataset.py ---

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import ast
import random
import math
from collections import Counter, defaultdict
import traceback

print("\n--- Anomaly Detection Dataset Setup ---")

# --- Parameters (from readme.md and graph_2.py) ---
paths_str = '''{ './dataset/11_REFIT_B2_WM_healthy_activations.npz': 'Healthy' }'''
try:
    NPZ_FILE_PATHS = ast.literal_eval(paths_str)
    if not isinstance(NPZ_FILE_PATHS, dict): raise TypeError("Paths not dict.")
except Exception as e:
    print(f"FATAL Error parsing paths: {e}")
    sys.exit(1)

SIGNAL_COLUMN_INDEX = 2 # "Appliance Energy" column
PADDING_VALUE = 0.0
RANDOM_SEED = 42

# Data split counts (Adjusted unhealthy counts slightly to match readme's val/test description exactly)
SPLIT_COUNTS = {
    'train': {'Healthy': 180, 'Unhealthy': 0},
    'val':   {'Healthy': 40,  'Unhealthy': 0},
    'test':  {'Healthy': 0,   'Unhealthy': 0}
}

# Label mapping
LABEL_MAP = {'Healthy': 0}
# Assign 1 to all unhealthy types for binary anomaly detection
# for i, label in enumerate(set(NPZ_FILE_PATHS.values()) - {'Healthy'}, start=1):
#      LABEL_MAP[label] = 1 # Treat all unhealthy as class 1
print(f"Using Label Map for Healthy-Only Training: {LABEL_MAP}")


# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Function Definitions ---

def load_and_pad_signals(paths_dict, column_index, padding_value):
    """
    Loads signals from multiple NPZ files, extracts a specific column,
    pads them to the maximum length, and collects metadata.

    Args:
        paths_dict (dict): Dictionary mapping NPZ file paths to origin labels.
        column_index (int): Index of the column containing the signal.
        padding_value (float): Value used for padding shorter sequences.

    Returns:
        tuple: (padded_signals_array, all_metadata, max_len)
               - np.ndarray: Padded signals [num_samples, max_len].
               - list: Metadata dictionaries for each signal.
               - int: Maximum sequence length before padding.
    """
    print(f"\n--- Loading & Padding Signals (Column {column_index}) ---")
    all_signals_list = []
    all_metadata = []
    total_files_processed = 0
    total_signals_extracted = 0
    global_idx_counter = 0

    if not paths_dict:
        raise ValueError("Input paths dictionary is empty.")

    for filepath, origin_label in paths_dict.items():
        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue

        current_file_signals = 0
        try:
            with np.load(filepath, allow_pickle=True) as data:
                keys = list(data.keys())
                if not keys:
                    print(f"  Info: File empty or unreadable: {filepath}")
                    continue

                total_files_processed += 1
                for key in keys:
                    try:
                        array = data[key]
                        # Check if array is suitable (2D, non-empty, has the column)
                        if isinstance(array, np.ndarray) and array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > column_index:
                            signal = array[:, column_index].astype(np.float32).copy()
                            if signal.size == 0:
                                print(f"    Warn: Empty signal for key '{key}' in {filepath}")
                                continue # Skip empty signals

                            all_signals_list.append(signal)
                            metadata = {'source_file': os.path.basename(filepath),
                                        'source_key': key,
                                        'label': origin_label, # Original detailed label
                                        'original_length': len(signal),
                                        'global_index': global_idx_counter}
                            all_metadata.append(metadata)
                            total_signals_extracted += 1
                            current_file_signals += 1
                            global_idx_counter += 1
                        else:
                            # Handle cases where data might be 1D if column_index is 0? (As in graph_2.py)
                             if isinstance(array, np.ndarray) and array.ndim == 1 and array.shape[0] > 0 and column_index == 0:
                                signal = array.astype(np.float32).copy()
                                all_signals_list.append(signal)
                                metadata = {'source_file': os.path.basename(filepath),
                                        'source_key': key,
                                        'label': origin_label, # Original detailed label
                                        'original_length': len(signal),
                                        'global_index': global_idx_counter}
                                all_metadata.append(metadata)
                                total_signals_extracted += 1
                                current_file_signals += 1
                                global_idx_counter += 1
                             else:
                                print(f"    Warn: Skipping key '{key}' in {filepath}. Unexpected data format: ndim={array.ndim if isinstance(array, np.ndarray) else 'N/A'}, shape={array.shape if isinstance(array, np.ndarray) else 'N/A'}")


                    except Exception as inner_e:
                        print(f"    Error processing key '{key}' in {filepath}: {inner_e}")
                        continue # Skip problematic keys

            print(f"  Processed file: {filepath} - Extracted {current_file_signals} signals.")

        except Exception as e:
            print(f"  Error processing file {filepath}: {e}")
            traceback.print_exc()
            if filepath in paths_dict: # Decrement count only if it was initially counted
                 total_files_processed -=1
            continue # Skip problematic files

    print(f"\n--- Load Summary ---")
    print(f"Processed {total_files_processed}/{len(paths_dict)} files.")
    print(f"Extracted {total_signals_extracted} signals in total.")

    if not all_signals_list:
        raise ValueError("FATAL Error: No valid signals extracted from any file.")

    print("\n--- Padding Signals ---")
    all_original_lengths = [m['original_length'] for m in all_metadata]
    max_len = max(all_original_lengths) if all_original_lengths else 0

    if max_len == 0:
        raise ValueError("FATAL Error: Max signal length is 0. Cannot pad.")

    print(f"Max signal length found: {max_len}")

    # Pad using numpy.pad for potentially better efficiency
    padded_signals_list = [np.pad(s, (0, max_len - len(s)), 'constant', constant_values=padding_value) for s in all_signals_list]
    padded_signals_array = np.stack(padded_signals_list).astype(np.float32)

    print(f"Shape of final padded signals array: {padded_signals_array.shape}")
    print("--- Loading & Padding Done ---")
    return padded_signals_array, all_metadata, max_len


def split_data_indices(all_metadata, split_counts, random_seed=RANDOM_SEED):
    """
    Splits the data indices into train, validation, and test sets based on counts
    and labels, ensuring representation from different unhealthy types.

    Args:
        all_metadata (list): List of metadata dictionaries for all samples.
        split_counts (dict): Dictionary defining the number of samples per label
                             type for each split ('train', 'val', 'test').
        random_seed (int): Seed for random sampling.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
               Lists of global indices for each split.
    """
    print("\n--- Splitting Data Indices ---")
    random.seed(random_seed) # Ensure reproducibility for splitting

    healthy_indices = [i for i, d in enumerate(all_metadata) if d['label'] == 'Healthy']
    unhealthy_indices_grouped = defaultdict(list)
    active_unhealthy_labels = set()
    for i, d in enumerate(all_metadata):
        if d['label'] != 'Healthy':
            unhealthy_indices_grouped[d['label']].append(i)
            active_unhealthy_labels.add(d['label'])

    print(f"Total Healthy samples: {len(healthy_indices)}")
    for label, indices in unhealthy_indices_grouped.items():
        print(f"Total {label} samples: {len(indices)}")

    active_unhealthy_labels = sorted(list(active_unhealthy_labels))
    num_active_unhealthy = len(active_unhealthy_labels)

    train_indices, val_indices, test_indices = [], [], []
    used_global_indices = set()

    for split_name, counts in split_counts.items():
        print(f"\nCreating {split_name} split indices...")
        current_split_indices = []

        # --- Healthy Samples ---
        n_healthy_requested = counts['Healthy']
        available_healthy = [idx for idx in healthy_indices if idx not in used_global_indices]
        n_healthy_actual = min(n_healthy_requested, len(available_healthy))

        if n_healthy_actual > 0:
            sampled_healthy = random.sample(available_healthy, n_healthy_actual)
            current_split_indices.extend(sampled_healthy)
            used_global_indices.update(sampled_healthy)
            print(f"  Added {len(sampled_healthy)} Healthy samples (requested {n_healthy_requested}, available {len(available_healthy)}).")
        else:
            print(f"  No available/requested Healthy samples added for {split_name}.")
        if n_healthy_actual < n_healthy_requested:
             print(f"  Warning: Could only add {n_healthy_actual} Healthy samples, requested {n_healthy_requested}.")


        # --- Unhealthy Samples ---
        n_unhealthy_total_requested = counts['Unhealthy']
        added_unhealthy_count = 0

        if num_active_unhealthy > 0 and n_unhealthy_total_requested > 0:
            # Calculate target per unhealthy type, ensuring we don't exceed total request
            n_per_type_target = math.ceil(n_unhealthy_total_requested / num_active_unhealthy)
            print(f"  Targeting ~{n_per_type_target} samples per unhealthy type (total unhealthy requested: {n_unhealthy_total_requested}).")

            # Distribute sampling across types
            unhealthy_samples_to_add = []
            for label in active_unhealthy_labels:
                available_unhealthy_type = [idx for idx in unhealthy_indices_grouped[label] if idx not in used_global_indices]
                n_to_sample_type = min(n_per_type_target, len(available_unhealthy_type))

                if n_to_sample_type > 0:
                    sampled_unhealthy_type = random.sample(available_unhealthy_type, n_to_sample_type)
                    unhealthy_samples_to_add.extend(sampled_unhealthy_type)
                    print(f"    Initially sampled {len(sampled_unhealthy_type)} '{label}' samples (available: {len(available_unhealthy_type)}).")
                else:
                     print(f"    No available unused '{label}' samples to sample.")

            # Shuffle the pooled unhealthy samples
            random.shuffle(unhealthy_samples_to_add)

            # Take exactly the required number
            n_unhealthy_final = min(n_unhealthy_total_requested, len(unhealthy_samples_to_add))
            final_sampled_unhealthy = unhealthy_samples_to_add[:n_unhealthy_final]

            current_split_indices.extend(final_sampled_unhealthy)
            used_global_indices.update(final_sampled_unhealthy)
            added_unhealthy_count = len(final_sampled_unhealthy)
            print(f"  Added {added_unhealthy_count} Unhealthy samples in total (requested {n_unhealthy_total_requested}).")
            if added_unhealthy_count < n_unhealthy_total_requested:
                print(f"  Warning: Could only add {added_unhealthy_count} Unhealthy samples, requested {n_unhealthy_total_requested}.")

        else:
            print(f"  No unhealthy samples requested or no unhealthy types available for {split_name}.")

        # Shuffle the final list for the split
        random.shuffle(current_split_indices)

        # Assign to the correct split list
        if split_name == 'train':
            train_indices = current_split_indices
        elif split_name == 'val':
            val_indices = current_split_indices
        else: # test
            test_indices = current_split_indices

        # Report final counts for the split
        split_labels_detailed = [all_metadata[idx]['label'] for idx in current_split_indices]
        split_labels_binary = ['Healthy' if l == 'Healthy' else 'Unhealthy' for l in split_labels_detailed]
        print(f"  {split_name} final counts (Detailed): {dict(Counter(split_labels_detailed))}")
        print(f"  {split_name} final counts (Binary):   {dict(Counter(split_labels_binary))}")
        print(f"  Total indices in {split_name}: {len(current_split_indices)}")

    print("\n--- Data Splitting Summary ---")
    print(f"Total Train indices: {len(train_indices)}")
    print(f"Total Val indices:   {len(val_indices)}")
    print(f"Total Test indices:  {len(test_indices)}")

    # Verification: Check for overlap
    assert len(set(train_indices).intersection(set(val_indices))) == 0, "Overlap detected between train and val!"
    assert len(set(train_indices).intersection(set(test_indices))) == 0, "Overlap detected between train and test!"
    assert len(set(val_indices).intersection(set(test_indices))) == 0, "Overlap detected between val and test!"
    print("Verified: No overlap between splits.")
    print("--- Data Splitting Done ---")

    return train_indices, val_indices, test_indices


class ApplianceDataset(Dataset):
    """PyTorch Dataset for appliance energy signals."""
    def __init__(self, signals, metadata, indices, label_map):
        """
        Args:
            signals (np.ndarray): Padded and potentially scaled signals [N_total, seq_len].
            metadata (list): List of metadata dictionaries for all signals.
            indices (list): List of global indices belonging to this split.
            label_map (dict): Dictionary mapping original string labels to numerical binary labels.
        """
        self.signals = signals[indices] # Select only signals for this split
        self.metadata_split = [metadata[i] for i in indices]
        self.indices = indices # Store the original global indices
        self.label_map = label_map

        # Add channel dimension: [num_samples_split, seq_len, 1]
        self.signals = np.expand_dims(self.signals, axis=-1)

        print(f"Created Dataset split with {len(self.indices)} samples. Signal shape: {self.signals.shape}")


    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (signal, label, global_index) from the dataset.

        Args:
            idx (int): Index of the sample within this split.

        Returns:
            tuple: (signal_tensor, label_tensor, global_index)
                   - torch.Tensor: Signal data [seq_len, 1].
                   - torch.Tensor: Binary label (0 or 1).
                   - int: The original global index of the sample.
        """
        signal = self.signals[idx]
        meta = self.metadata_split[idx]
        original_label = meta['label']
        global_index = meta['global_index']

        # Map original label to binary numerical label (0: Healthy, 1: Unhealthy)
        numerical_label = self.label_map.get(original_label, -1) # Default to -1 if label not found
        if numerical_label == -1:
             print(f"Warning: Unknown label '{original_label}' encountered for global index {global_index}. Assigning -1.")


        # Convert to PyTorch tensors
        # Signal shape becomes [seq_len, input_dim=1]
        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.tensor(numerical_label, dtype=torch.long) # Use long for CrossEntropyLoss, float for BCE

        return signal_tensor, label_tensor, global_index


def fit_scaler(padded_signals, train_indices, padding_value=0.0):
    """
    Fits a StandardScaler only on the non-padded parts of the training data.

    Args:
        padded_signals (np.ndarray): Padded signals [N_total, seq_len].
        train_indices (list): List of global indices for the training set.
        padding_value (float): The value used for padding.

    Returns:
        sklearn.preprocessing.StandardScaler: The fitted scaler object.
    """
    print("\n--- Fitting Scaler on Training Data (Non-Padded Values) ---")
    train_signals = padded_signals[train_indices]
    # Extract non-padded values from training data
    # Flatten the sequences and filter out padding
    non_padded_values = train_signals[train_signals != padding_value].reshape(-1, 1)

    if non_padded_values.size == 0:
        print("Warning: No non-padded values found in training data to fit scaler. Returning unfitted scaler.")
        return StandardScaler() # Return unfitted scaler

    scaler = StandardScaler()
    scaler.fit(non_padded_values)
    print(f"Scaler fitted with mean: {scaler.mean_[0]:.4f}, scale: {scaler.scale_[0]:.4f}")
    print("--- Scaler Fitting Done ---")
    return scaler

def scale_data(padded_signals, scaler, padding_value=0.0):
    """
    Applies a pre-fitted StandardScaler to the signals, handling padding.

    Args:
        padded_signals (np.ndarray): Padded signals [N, seq_len].
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler object.
        padding_value (float): The value used for padding.

    Returns:
        np.ndarray: Scaled signals [N, seq_len].
    """
    print("\n--- Scaling Data ---")
    scaled_signals = np.copy(padded_signals) # Avoid modifying original data

    # Iterate through each signal to scale only non-padded parts
    for i in range(scaled_signals.shape[0]):
        signal = scaled_signals[i]
        non_padded_mask = (signal != padding_value)
        if np.any(non_padded_mask): # Check if there are any non-padded values
            signal[non_padded_mask] = scaler.transform(signal[non_padded_mask].reshape(-1, 1)).flatten()

    # Handle potential NaNs/Infs introduced by scaling (e.g., if scale_ is near zero)
    scaled_signals = np.nan_to_num(scaled_signals, nan=padding_value, posinf=padding_value, neginf=padding_value)
    print("--- Scaling Done ---")
    return scaled_signals


def get_dataloaders(batch_size, signal_col_idx=SIGNAL_COLUMN_INDEX, scale=True, num_workers=0):
    """
    Main function to load, preprocess, split data, and create DataLoaders.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        signal_col_idx (int): Index of the signal column to extract.
        scale (bool): Whether to apply StandardScaler to the signals.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: Contains:
            - train_loader (DataLoader): DataLoader for training data.
            - val_loader (DataLoader): DataLoader for validation data.
            - test_loader (DataLoader): DataLoader for testing data.
            - all_metadata (list): Metadata for all samples.
            - train_indices (list): Global indices for training split.
            - val_indices (list): Global indices for validation split.
            - test_indices (list): Global indices for testing split.
            - max_len (int): Maximum sequence length.
            - scaler (StandardScaler or None): Fitted scaler object if scale=True, else None.
    """
    # 1. Load and Pad
    padded_signals, all_metadata, max_len = load_and_pad_signals(
        NPZ_FILE_PATHS, signal_col_idx, PADDING_VALUE
    )

    # 2. Split Indices
    train_indices, val_indices, test_indices = split_data_indices(
        all_metadata, SPLIT_COUNTS, RANDOM_SEED
    )

    # 3. Scaling (Optional)
    scaler = None
    if scale:
        scaler = fit_scaler(padded_signals, train_indices, PADDING_VALUE)
        signals_to_use = scale_data(padded_signals, scaler, PADDING_VALUE)
    else:
        signals_to_use = padded_signals
        print("\n--- Skipping Scaling ---")


    # 4. Create Datasets
    print("\n--- Creating Datasets ---")
    train_dataset = ApplianceDataset(signals_to_use, all_metadata, train_indices, LABEL_MAP)
    val_dataset = ApplianceDataset(signals_to_use, all_metadata, val_indices, LABEL_MAP)
    test_dataset = ApplianceDataset(signals_to_use, all_metadata, test_indices, LABEL_MAP)
    print("--- Datasets Created ---")

    # 5. Create DataLoaders
    print("\n--- Creating DataLoaders ---")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader:   {len(val_loader)} batches")
    print(f"Test loader:  {len(test_loader)} batches")
    print("--- DataLoaders Created ---")

    return (train_loader, val_loader, test_loader,
            all_metadata, train_indices, val_indices, test_indices,
            max_len, scaler)

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    print("\n--- Running dataset.py directly for testing ---")
    BATCH_SIZE = 32
    try:
        (train_loader, val_loader, test_loader,
         all_metadata, train_idx, val_idx, test_idx,
         max_len, scaler) = get_dataloaders(batch_size=BATCH_SIZE, scale=True)

        print(f"\n--- Example Batch from Train Loader ---")
        for signals, labels, global_indices in train_loader:
            print(f"Signals batch shape: {signals.shape}") # Should be [batch_size, max_len, 1]
            print(f"Labels batch shape: {labels.shape}")   # Should be [batch_size]
            print(f"Labels batch example: {labels[:5]}")
            print(f"Global indices batch shape: {global_indices.shape}")
            print(f"Global indices example: {global_indices[:5]}")
            break # Only show one batch

        print(f"\nTotal metadata entries: {len(all_metadata)}")
        print(f"Max sequence length: {max_len}")
        print(f"Scaler object: {'Fitted' if scaler else 'None'}")
        print("\n--- dataset.py Test Finished Successfully ---")

    except FileNotFoundError as fnf_error:
        print(f"\nFATAL ERROR: File Not Found. {fnf_error}")
        print("Please ensure the 'microwave-dataset' folder exists in the same directory and contains the required NPZ files.")
    except ValueError as val_error:
        print(f"\nFATAL ERROR: Value Error. {val_error}")
        traceback.print_exc()
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred.")
        traceback.print_exc()

# --- END OF FILE dataset.py ---