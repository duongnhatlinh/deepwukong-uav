from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GatedGraphConv, GlobalAttention, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn as nn 
import torch.nn.functional as F
from src.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder


class GraphConvEncoder(torch.nn.Module):
    """
    Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    (https://arxiv.org/pdf/1609.02907.pdf)
    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        
        self.input_GCL = GCNConv(config.rnn.hidden_size, config.hidden_size)
        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)
        
        # Initialize hidden layers
        self.hidden_GCLs = nn.ModuleList()
        self.hidden_GPLs = nn.ModuleList()

        for i in range(config.n_hidden_layers - 1):
            self.hidden_GCLs.append(GCNConv(config.hidden_size, config.hidden_size))
            self.hidden_GPLs.append(TopKPooling(config.hidden_size, ratio=config.pooling_ratio))
        
        # self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))
        self.attpool = AttentionalAggregation(gate_nn=nn.Linear(config.hidden_size, 1))


    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        
        # First layer
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(
            node_embedding, edge_index, None, batch
        )
        
        # Use global mean pooling instead of attention
        out = self.attpool(node_embedding, batch)

        # Hidden layers
        for gcl, gpl in zip(self.hidden_GCLs, self.hidden_GPLs):
            node_embedding = F.relu(gcl(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = gpl(
                node_embedding, edge_index, None, batch
            )
            out += self.attpool(node_embedding, batch)

        return out
class GatedGraphConvEncoder(torch.nn.Module):
    """
    from Li et al.: Gated Graph Sequence Neural Networks (ICLR 2016)
    (https://arxiv.org/pdf/1511.05493.pdf)
    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GatedGraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru)
        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)
        
        # Initialize hidden layers - MUST use GatedGraphConv!
        self.hidden_GCLs = nn.ModuleList()
        self.hidden_GPLs = nn.ModuleList()

        for i in range(config.n_hidden_layers - 1):

            self.hidden_GCLs.append(
                GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru)
            )
            self.hidden_GPLs.append(
                TopKPooling(config.hidden_size, ratio=config.pooling_ratio)
            )

        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))
        
    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        
        # First layer
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(
            node_embedding, edge_index, None, batch
        )
        
        # Use global mean pooling instead of attention
        out = self.attpool(node_embedding, batch)

        # Hidden layers
        for gcl, gpl in zip(self.hidden_GCLs, self.hidden_GPLs):
            node_embedding = F.relu(gcl(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = gpl(
                node_embedding, edge_index, None, batch
            )
            out += self.attpool(node_embedding, batch)

        return out










# ------------------------------------------
# Thêm vào src/models/modules/gnns.py

class ResidualGCNLayer(torch.nn.Module):
    """Residual GCN Layer with skip connections"""
    def __init__(self, hidden_size):
        super().__init__()
        self.gcn = GCNConv(hidden_size, hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        residual = x
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x + residual  # Skip connection

class AdaptiveGraphPooling(torch.nn.Module):
    """Adaptive pooling for graphs"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_pool = GlobalAttention(nn.Linear(hidden_size, 1))
        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool
        self.combine = nn.Linear(hidden_size * 3, hidden_size)
        
    def forward(self, x, batch):
        # Multiple pooling strategies
        att_pool = self.attention_pool(x, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        # Combine different pooling results
        combined = torch.cat([att_pool, mean_pool, max_pool], dim=1)
        return self.combine(combined)

class EnhancedGraphConvEncoder(torch.nn.Module):
    """Enhanced GCN with multi-scale features and attention"""
    
    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int, pad_idx: int):
        super().__init__()
        self.__config = config
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)
        
        # Multi-scale convolutions
        self.local_conv = GCNConv(config.rnn.hidden_size, config.hidden_size//2)
        self.global_conv = GCNConv(config.rnn.hidden_size, config.hidden_size//2)
        
        # Cross-attention between scales  
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Residual GCN layers
        self.residual_layers = nn.ModuleList([
            ResidualGCNLayer(config.hidden_size) 
            for _ in range(config.n_hidden_layers - 1)  # -1 vì đã có local/global conv
        ])
        
        # Enhanced pooling
        self.adaptive_pool = AdaptiveGraphPooling(config.hidden_size)
        
        # Batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(config.drop_out)
        
    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden] - Node embeddings
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        
        # Multi-scale feature extraction
        local_features = F.relu(self.local_conv(node_embedding, edge_index))
        global_features = F.relu(self.global_conv(node_embedding, edge_index))
        
        # Combine multi-scale features
        combined_features = torch.cat([local_features, global_features], dim=1)
        
        # Apply cross-attention (reshape for attention)
        if combined_features.size(0) > 0:
            # Group by batch for attention
            attended_features = combined_features
            for batch_id in torch.unique(batch):
                mask = batch == batch_id
                if mask.sum() > 1:  # Need at least 2 nodes for attention
                    batch_features = combined_features[mask].unsqueeze(0)
                    attended_batch, _ = self.cross_attention(
                        batch_features, batch_features, batch_features
                    )
                    attended_features[mask] = attended_batch.squeeze(0)
            
            combined_features = attended_features
        
        # Apply residual layers
        for residual_layer in self.residual_layers:
            combined_features = residual_layer(combined_features, edge_index)
        
        # Enhanced pooling
        graph_representation = self.adaptive_pool(combined_features, batch)
        
        # Final normalization and dropout
        graph_representation = self.batch_norm(graph_representation)
        graph_representation = self.dropout(graph_representation)
        
        return graph_representation