"""
Graph Neural Network (GNN) Pitch Control Feature module.

This module constructs an adjacency matrix from passing/pressing event data to represent
team passing networks and implements a Graph Attention Network (GAT) layer to output
an Expected Threat (xT) Network Dominance scalar feature.

### Mathematical Formulation

#### 1. Graph Construction
Let the graph be defined as $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ is the
set of nodes representing players (or zones) and $\mathcal{E}$ is the set of edges representing
passes between players. The adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ is defined by
the passing frequencies or success rates between players $i$ and $j$, where $N = |\mathcal{V}|$.

#### 2. Node Features
Each node $i$ has an associated feature vector $\mathbf{h}_i \in \mathbb{R}^{F}$, where $F$ is the number
of features (e.g., player position coordinates, historical xT, pressing pressure).

#### 3. Graph Attention Layer (GAT)
The GAT layer computes a single attention coefficient $e_{ij}$ between nodes $i$ and $j$ as:
$$ e_{ij} = \text{LeakyReLU}\left( \mathbf{a}^T [ \mathbf{W}\mathbf{h}_i \,||\, \mathbf{W}\mathbf{h}_j ] \right) $$
where $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a learnable weight matrix, $\mathbf{a} \in \mathbb{R}^{2F'}$
is a learnable attention weight vector, and $||$ represents concatenation.

These coefficients are normalized across all neighbors $j \in \mathcal{N}_i$ using the softmax function:
$$ \alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})} $$

The updated node features $\mathbf{h}'_i$ are then computed as a linear combination of neighbor features:
$$ \mathbf{h}'_i = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right) $$
where $\sigma$ is an activation function (e.g., ELU).

#### 4. Expected Threat (xT) Network Dominance
The final scalar feature for the network (or a specific subgraph) is computed by aggregating
the node features (e.g., via global mean pooling) and passing them through an MLP:
$$ \text{xT\_Dominance} = \text{MLP} \left( \frac{1}{N} \sum_{i=1}^N \mathbf{h}'_i \right) $$
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Dict, Any


class ExpectedThreatGAT(nn.Module):
    """
    Graph Attention Network to compute Expected Threat (xT) Network Dominance.

    This network processes player passing networks where nodes are players, edges are passes,
    and node features represent player context (e.g., location, pressing pressure).

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features in the GAT layer.
        out_channels (int): Number of output features per node from GAT layer.
        heads (int): Number of attention heads for the GAT layer.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        super(ExpectedThreatGAT, self).__init__()
        self.dropout = dropout

        # Graph Attention Layer
        # Computes attention coefficients: \alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j]))}{\sum_{k} \dots}
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        
        # Second GAT layer to consolidate multi-head output
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        # Multi-Layer Perceptron for final scalar output (xT Network Dominance)
        # \text{xT\_Dominance} = \text{MLP}(\text{GlobalMeanPool}(\mathbf{H}'))
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GAT network.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph edge connectivity of shape [2, num_edges].
            batch (torch.Tensor): Batch vector indicating which graph each node belongs to, of shape [num_nodes].

        Returns:
            torch.Tensor: Expected Threat (xT) Network Dominance scalar for each graph in the batch, shape [batch_size, 1].
        """
        # Node embeddings
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        
        # Global aggregation to form a graph-level embedding
        x_graph = global_mean_pool(x, batch)

        # Compute scalar dominance score
        xt_dominance = self.mlp(x_graph)
        return xt_dominance


def construct_passing_network(events: List[Dict[str, Any]]) -> Data:
    """
    Constructs a PyTorch Geometric Data object from raw event data.

    The event data represents hypothetical passing and pressing events.
    Each event dictionary should contain:
        - 'passer_id': int
        - 'receiver_id': int
        - 'pass_success': bool
        - 'passer_location': Tuple[float, float]
        - 'pressing_intensity': float

    Args:
        events (List[Dict[str, Any]]): List of event dictionaries.

    Returns:
        Data: PyTorch Geometric Data object containing `x` (node features), `edge_index` (connectivity),
              and optionally `edge_attr` (edge weights).
    """
    # Extract unique player IDs involved in the events
    player_ids = set()
    for event in events:
        if 'passer_id' in event:
            player_ids.add(event['passer_id'])
        if 'receiver_id' in event:
            player_ids.add(event['receiver_id'])
    
    player_list = sorted(list(player_ids))
    player_to_idx = {pid: idx for idx, pid in enumerate(player_list)}
    num_nodes = len(player_list)
    
    # Initialize node features and edge structures
    # Node features: [x_coord, y_coord, avg_pressing_intensity, pass_attempts]
    node_features = np.zeros((num_nodes, 4), dtype=np.float32)
    edge_src = []
    edge_dst = []
    
    for event in events:
        if 'passer_id' in event and 'receiver_id' in event:
            src_idx = player_to_idx[event['passer_id']]
            dst_idx = player_to_idx[event['receiver_id']]
            
            # Add edge if pass is successful (could also include unsuccessful with lower weight)
            if event.get('pass_success', True):
                edge_src.append(src_idx)
                edge_dst.append(dst_idx)
            
            # Aggregate node features based on passer
            if 'passer_location' in event:
                loc_x, loc_y = event['passer_location']
                node_features[src_idx, 0] += loc_x
                node_features[src_idx, 1] += loc_y
            
            if 'pressing_intensity' in event:
                node_features[src_idx, 2] += event['pressing_intensity']
            
            node_features[src_idx, 3] += 1.0

    # Average node features where applicable (e.g., locations, pressing intensity)
    for i in range(num_nodes):
        attempts = node_features[i, 3]
        if attempts > 0:
            node_features[i, 0] /= attempts
            node_features[i, 1] /= attempts
            node_features[i, 2] /= attempts
            
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    
    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Return as PyTorch Geometric Data object
    return Data(x=x, edge_index=edge_index)


if __name__ == "__main__":
    # Example usage
    hypothetical_events = [
        {'passer_id': 10, 'receiver_id': 8, 'pass_success': True, 'passer_location': (50.0, 30.0), 'pressing_intensity': 0.2},
        {'passer_id': 8, 'receiver_id': 9, 'pass_success': True, 'passer_location': (60.0, 40.0), 'pressing_intensity': 0.5},
        {'passer_id': 9, 'receiver_id': 10, 'pass_success': True, 'passer_location': (70.0, 35.0), 'pressing_intensity': 0.8},
        {'passer_id': 10, 'receiver_id': 11, 'pass_success': False, 'passer_location': (75.0, 30.0), 'pressing_intensity': 0.9},
    ]

    # Construct the graph
    graph_data = construct_passing_network(hypothetical_events)
    print("Graph constructed:")
    print(f"Num nodes: {graph_data.num_nodes}")
    print(f"Num edges: {graph_data.num_edges}")
    print(f"Node features shape: {graph_data.x.shape}")
    print(f"Edge index shape: {graph_data.edge_index.shape}")

    # Initialize model
    model = ExpectedThreatGAT(in_channels=4, hidden_channels=16, out_channels=8, heads=2)
    model.eval()

    # Create dummy batch index (all nodes belong to graph 0)
    batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)

    # Forward pass
    with torch.no_grad():
        xt_dominance = model(graph_data.x, graph_data.edge_index, batch)
    
    print(f"Expected Threat (xT) Network Dominance:\\n{xt_dominance.item():.4f}")
