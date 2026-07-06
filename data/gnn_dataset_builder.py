"""
GNN Dataset Builder for Football Analytics.

This module provides the :class:`GNNDatasetBuilder` which transforms standard 
tabular football match data into PyTorch Geometric (PyG) graphs.

The transformation models the data as a Temporal Graph where:
- Nodes represent the teams.
- Edges represent matches played between teams.
- Node features contain a team's rolling statistics (e.g., xG, possession).
- Edge features contain match-specific statistics or outcomes.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Optional

class GNNDatasetBuilder:
    """
    A class to build a PyTorch Geometric dataset from tabular match data.
    
    This class converts standard football spreadsheet data (e.g., team stats, 
    possession, expected goals) into a sequence of PyTorch Geometric ``Data`` objects 
    representing a Temporal Graph.
    
    Mathematical Formulation:
    -------------------------
    Let :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{E})` be a graph where:
    - :math:`\mathcal{V}` represents the set of nodes (Teams).
    - :math:`\mathcal{E}` represents the set of edges (Matches between teams).
    - :math:`\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times F_v}` represents the node feature matrix 
      (e.g., rolling averages of xG, sprints, possession).
    - :math:`\mathbf{E} \in \mathbb{R}^{|\mathcal{E}| \times F_e}` represents the edge feature matrix 
      (e.g., match-specific statistics or outcomes).
      
    For each time step (or round) :math:`t`, we construct a graph :math:`\mathcal{G}_t`, ensuring 
    that the temporal dynamics of the tournament are captured.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        team_col: str, 
        opp_col: str, 
        date_col: str, 
        node_feature_cols: List[str], 
        edge_feature_cols: List[str], 
        target_col: Optional[str] = None
    ):
        """
        Initializes the builder with the dataset and column mappings.
        
        Args:
            df (pd.DataFrame): The tabular dataset containing match data.
            team_col (str): The column representing the home/primary team.
            opp_col (str): The column representing the away/opponent team.
            date_col (str): The column representing the date or round of the match.
            node_feature_cols (List[str]): Columns to be used as node features.
            edge_feature_cols (List[str]): Columns to be used as edge features.
            target_col (Optional[str]): The column representing the target variable (e.g., match outcome).
        """
        self.df = df.copy()
        self.team_col = team_col
        self.opp_col = opp_col
        self.date_col = date_col
        self.node_feature_cols = node_feature_cols
        self.edge_feature_cols = edge_feature_cols
        self.target_col = target_col
        
        # Ensure chronological ordering
        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df = self.df.sort_values(by=self.date_col).reset_index(drop=True)
            
        # Create a mapping from team name to an integer ID
        unique_teams = pd.concat([self.df[self.team_col], self.df[self.opp_col]]).unique()
        self.team_to_id = {team: idx for idx, team in enumerate(unique_teams)}
        self.num_nodes = len(unique_teams)

    def _get_node_features(self, current_date: pd.Timestamp) -> torch.Tensor:
        """
        Computes the node feature matrix :math:`\mathbf{X}` for a given point in time.
        
        Retrieves the most recent stats (or computes rolling averages) for each team 
        prior to or on the given date.
        
        Args:
            current_date (pd.Timestamp): The date up to which data is considered.
            
        Returns:
            torch.Tensor: A tensor of shape :math:`(|\mathcal{V}|, F_v)` containing node features.
        """
        # Filter dataframe up to the current date
        historical_df = self.df[self.df[self.date_col] <= current_date]
        
        # Initialize node features matrix with zeros
        x = np.zeros((self.num_nodes, len(self.node_feature_cols)))
        
        for team, team_id in self.team_to_id.items():
            # Get historical stats where the team was the primary team
            team_stats = historical_df[historical_df[self.team_col] == team]
            if not team_stats.empty:
                # Use the most recent stats as node features
                latest_stats = team_stats.iloc[-1][self.node_feature_cols].values
                # Handle possible NaN values by filling with 0
                x[team_id] = np.nan_to_num(latest_stats.astype(np.float32))
                
        return torch.tensor(x, dtype=torch.float32)

    def build_dataset(self) -> List[Data]:
        """
        Constructs the sequence of PyTorch Geometric graphs.
        
        Let :math:`T` be the number of unique time steps (e.g., match days). 
        For each :math:`t \in T`, a graph :math:`\mathcal{G}_t` is constructed where nodes 
        are updated with the latest available features, and edges represent the matches 
        occurring at time :math:`t`.
        
        Returns:
            List[Data]: A list of ``torch_geometric.data.Data`` objects, one for each time step.
        """
        data_list = []
        unique_dates = self.df[self.date_col].unique()
        
        for date in unique_dates:
            matches_on_date = self.df[self.df[self.date_col] == date]
            
            edge_indices = []
            edge_attrs = []
            targets = []
            
            for _, match in matches_on_date.iterrows():
                team_u = self.team_to_id[match[self.team_col]]
                team_v = self.team_to_id[match[self.opp_col]]
                
                # Add bidirectional edges for the match
                edge_indices.append([team_u, team_v])
                edge_indices.append([team_v, team_u])
                
                # Process edge features
                edge_feat = match[self.edge_feature_cols].values.astype(np.float32)
                edge_feat = np.nan_to_num(edge_feat)
                edge_attrs.append(edge_feat)
                edge_attrs.append(edge_feat)  # Symmetric feature for reverse edge
                
                # Process targets if provided
                if self.target_col:
                    targets.append(match[self.target_col])
                    targets.append(match[self.target_col])
                    
            if not edge_indices:
                continue
                
            # Convert to PyTorch tensors
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
            
            # Compute node features up to this date
            x = self._get_node_features(pd.Timestamp(date))
            
            # Construct PyG Data object
            kwargs = {
                'x': x,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'date': date
            }
            if self.target_col:
                kwargs['y'] = torch.tensor(targets, dtype=torch.float32)
                
            graph_data = Data(**kwargs)
            data_list.append(graph_data)
            
        return data_list
