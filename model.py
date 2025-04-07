# --- START OF FILE model.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PositionalEncoding # Import from utils.py
import math

# Check if torch_geometric is installed, provide informative error if not
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    GCNConv = None
    dense_to_sparse = None
    print("Warning: PyTorch Geometric not found. GCN functionality will be disabled.")
    print("Please install it following instructions at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")

class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network Encoder for learning node (cluster state) embeddings.
    Takes node features (cluster centroids) and adjacency matrix as input.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 8, dropout: float = 0.1, activation=F.relu):
        """
        Args:
            node_feature_dim (int): Dimension of input features for each node (15 in this case).
            hidden_dim (int): Dimension of hidden GCN layers.
            output_dim (int): Dimension of the output node embeddings.
            num_layers (int): Number of GCN layers (as per readme: 8).
            dropout (float): Dropout probability.
            activation: Activation function for hidden layers (e.g., F.relu).
        """
        super().__init__()
        if GCNConv is None:
            raise ImportError("PyTorch Geometric (GCNConv) is required for GCNEncoder.")

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.convs = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("Number of GCN layers must be at least 1.")

        # Input layer
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer (if num_layers > 1)
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else: # If only 1 layer, the first layer is the output layer
             self.convs[0] = GCNConv(node_feature_dim, output_dim) # Overwrite first layer


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the GCN Encoder.

        Args:
            x: Node features (cluster centroids), shape [num_nodes, node_feature_dim].
            edge_index: Graph connectivity in COO format, shape [2, num_edges].
            edge_weight: Edge weights corresponding to edge_index (optional for GCNConv, but can be used by variants). Shape [num_edges].

        Returns:
            torch.Tensor: Output node embeddings, shape [num_nodes, output_dim].
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight) # Pass edge_weight if needed by GCNConv variant
            if i < self.num_layers - 1: # Apply activation and dropout to all but the output layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # else: # Apply sigmoid to the last layer as specified in readme
            #     x = torch.sigmoid(x)
        # Using sigmoid on the final embedding might compress the information too much.
        # Let's stick with ReLU for hidden and no activation/sigmoid for the last layer initially,
        # unless sigmoid is explicitly required for a downstream task constraint.
        # Reverting to readme: Applying sigmoid to the *last layer output*.
        if self.num_layers > 0:
             x = torch.sigmoid(x)

        return x

class GCNTransformerAutoencoder(nn.Module):
    """
    GCN-Transformer Autoencoder for time-series reconstruction and anomaly detection.
    Encodes the input sequence using a Transformer and integrates graph-based
    state information from a GCN into the decoder's context.
    """
    def __init__(self,
                 input_dim: int,          # Dimension of input sequence elements (1 for univariate)
                 seq_len: int,            # Length of the input sequences
                 node_feature_dim: int,   # Dimension of GCN node features (15)
                 n_clusters: int,         # Number of nodes/clusters in the graph
                 d_model: int,            # Transformer embedding dimension
                 nhead: int,              # Number of attention heads in Transformer
                 num_encoder_layers: int, # Number of Transformer encoder layers
                 num_decoder_layers: int, # Number of Transformer decoder layers
                 dim_feedforward: int,    # Dimension of feedforward network in Transformer
                 gcn_hidden_dim: int,     # Hidden dimension for GCN layers
                 gcn_out_dim: int,        # Output dimension of GCN (state embedding dim)
                 gcn_layers: int = 8,     # Number of GCN layers (from readme)
                 dropout: float = 0.1):
        """
        Args:
            input_dim (int): Input feature dimension (usually 1).
            seq_len (int): Length of input sequences.
            node_feature_dim (int): Dimension of node features (15).
            n_clusters (int): Number of clusters/nodes in the graph.
            d_model (int): Internal dimension of the Transformer.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of Transformer encoder layers.
            num_decoder_layers (int): Number of Transformer decoder layers.
            dim_feedforward (int): Dimension of the Transformer feedforward layer.
            gcn_hidden_dim (int): Hidden dimension for GCN layers.
            gcn_out_dim (int): Output dimension for GCN encoder (state embedding).
            gcn_layers (int): Number of GCN layers.
            dropout (float): Dropout rate.
        """
        super().__init__()

        if GCNConv is None or dense_to_sparse is None:
            raise ImportError("PyTorch Geometric (GCNConv, dense_to_sparse) is required.")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.gcn_out_dim = gcn_out_dim
        self.n_clusters = n_clusters

        # --- GCN Encoder ---
        # Input: Node features [n_clusters, node_feature_dim=15], Adj matrix [n_clusters, n_clusters]
        self.gcn_encoder = GCNEncoder(node_feature_dim, gcn_hidden_dim, gcn_out_dim,
                                      num_layers=gcn_layers, dropout=dropout, activation=F.relu)

        # --- Transformer Components ---
        # 1. Input Embedding for the time series sequence
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len) # Max len should match seq_len

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 3. Projection layer to combine sequence memory and GCN state embedding
        # Input size = d_model (from seq encoder) + gcn_out_dim (from GCN state)
        decoder_memory_input_dim = d_model + gcn_out_dim
        self.memory_projection = nn.Linear(decoder_memory_input_dim, d_model) # Project combined features to d_model

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='relu')
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # 5. Output Layer (Linear for reconstruction)
        # Projects decoder output [batch_size, seq_len, d_model] back to input shape [batch_size, seq_len, input_dim=1]
        self.output_linear = nn.Linear(d_model, input_dim)

        # TODO: Add optional 1D Deconvolution layer as mentioned in readme?
        # Example (needs careful design based on desired upsampling/reconstruction strategy):
        # self.deconv_layers = nn.Sequential(...)
        # self.final_linear = nn.Linear(...)

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights for linear layers."""
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)
        self.memory_projection.bias.data.zero_()
        self.memory_projection.weight.data.uniform_(-initrange, initrange)
        # GCNConv layers have their own default initializations (Glorot/Xavier)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a square mask for the sequence. E.g., for sz=5:
        [[0., -inf, -inf, -inf, -inf],
         [0., 0., -inf, -inf, -inf],
         [0., 0., 0., -inf, -inf],
         [0., 0., 0., 0., -inf],
         [0., 0., 0., 0., 0.]]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                src: torch.Tensor,           # Input sequence windows: [batch_size, seq_len, input_dim=1]
                state_indices: torch.Tensor, # Cluster index for each window: [batch_size]
                node_features: torch.Tensor, # Node features (centroids): [n_clusters, node_feature_dim=15]
                adj_matrix: torch.Tensor,    # Adjacency matrix: [n_clusters, n_clusters]
                tgt: torch.Tensor = None,    # Target sequence for decoder (e.g., src for autoencoding)
                src_mask: torch.Tensor = None, # Optional mask for encoder
                tgt_mask: torch.Tensor = None  # Optional mask for decoder
                ) -> torch.Tensor:
        """
        Forward pass of the GCN-Transformer Autoencoder.

        Args:
            src (torch.Tensor): Input sequence [batch_size, seq_len, input_dim].
            state_indices (torch.Tensor): Node index for each sample [batch_size].
            node_features (torch.Tensor): Static node features [n_clusters, node_feature_dim].
            adj_matrix (torch.Tensor): Static adjacency matrix [n_clusters, n_clusters].
            tgt (torch.Tensor, optional): Target sequence for decoder [batch_size, seq_len, input_dim]. If None, uses src.
            src_mask (torch.Tensor, optional): Source sequence mask.
            tgt_mask (torch.Tensor, optional): Target sequence mask (causal mask generated if None).

        Returns:
            torch.Tensor: Reconstructed output sequence [batch_size, seq_len, input_dim].
        """
        batch_size = src.size(0)
        device = src.device

        # --- Prepare Graph Data ---
        # Ensure graph data is on the correct device
        node_features = node_features.to(device)
        adj_matrix = adj_matrix.to(device)
        # Convert adjacency matrix to edge_index format for PyG
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        if edge_weight is not None:
            edge_weight = edge_weight.float() # Ensure edge weights are float

        # --- GCN Encoder ---
        # Get embeddings for all node states
        # Output shape: [n_clusters, gcn_out_dim]
        all_state_embeddings = self.gcn_encoder(node_features, edge_index, edge_weight)

        # Select the state embedding corresponding to each window in the batch
        # Shape: [batch_size, gcn_out_dim]
        # Ensure state_indices are within bounds
        valid_indices = (state_indices >= 0) & (state_indices < self.n_clusters)
        if not torch.all(valid_indices):
            # Handle invalid indices (e.g., from padding or errors) - maybe assign a default embedding?
            print(f"Warning: Invalid state indices detected: {state_indices[~valid_indices]}. Max index: {self.n_clusters-1}")
            # Option: Clamp indices (might not be ideal)
            state_indices = torch.clamp(state_indices, 0, self.n_clusters - 1)
            # Option: Use a zero embedding or learnable default embedding for invalid states

        batch_state_embeddings = all_state_embeddings[state_indices]

        # --- Transformer Sequence Encoder ---
        # Embed the input sequence
        # Input shape: [batch_size, seq_len, input_dim] -> Output shape: [batch_size, seq_len, d_model]
        src_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        # Add positional encoding. Permute for PositionalEncoding expected shape [seq_len, batch_size, d_model]
        # Shape: [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        src_embedded = src_embedded.permute(1, 0, 2)
        src_embedded = self.pos_encoder(src_embedded)
        # Permute back to [batch_size, seq_len, d_model] for TransformerEncoder with batch_first=True
        src_embedded = src_embedded.permute(1, 0, 2)

        # Pass through Transformer Encoder
        # Output shape: [batch_size, seq_len, d_model]
        sequence_memory = self.transformer_encoder(src_embedded, src_mask)

        # --- Combine Sequence Memory and GCN State Embedding ---
        # Expand state embedding to match sequence length: [batch_size, gcn_out_dim] -> [batch_size, seq_len, gcn_out_dim]
        expanded_state_embeddings = batch_state_embeddings.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Concatenate along the feature dimension: [batch_size, seq_len, d_model + gcn_out_dim]
        combined_memory_features = torch.cat((sequence_memory, expanded_state_embeddings), dim=-1)

        # Project the combined features back to d_model for the decoder's memory input
        # Shape: [batch_size, seq_len, d_model]
        projected_memory = self.memory_projection(combined_memory_features)

        # --- Transformer Decoder ---
        if tgt is None:
            tgt = src # Use source as target for autoencoding reconstruction task

        # Embed target sequence
        # Shape: [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
        tgt_embedded = self.input_embed(tgt) * math.sqrt(self.d_model)
        # Add positional encoding: Permute -> Add PE -> Permute back
        # Shape: [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        tgt_embedded = tgt_embedded.permute(1, 0, 2)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        # Shape: [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
        tgt_embedded = tgt_embedded.permute(1, 0, 2)


        # Generate causal mask for the target sequence if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)

        # Pass target embedding and projected memory through Transformer Decoder
        # Output shape: [batch_size, seq_len, d_model]
        decoder_output = self.transformer_decoder(tgt_embedded, projected_memory, tgt_mask=tgt_mask)

        # --- Output Layer ---
        # Project decoder output back to the original input dimension
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, input_dim]
        output = self.output_linear(decoder_output)

        return output


# --- END OF FILE model.py ---