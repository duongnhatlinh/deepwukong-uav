import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TopKPooling, GCNConv, GatedGraphConv, GlobalAttention
from src.vocabulary import Vocabulary
from src.utils import PAD, UNK


def create_debug_config():
    """Tạo config cho debug GNN"""
    return DictConfig({
        'embed_size': 64,
        'hidden_size': 32,  # GNN hidden size
        'n_hidden_layers': 3,
        'n_gru': 3,  # For GatedGraphConv
        'pooling_ratio': 0.8,
        'rnn': {
            'hidden_size': 64,
            'num_layers': 1,
            'use_bi': True,
            'drop_out': 0.1
        }
    })

def create_debug_vocab():
    """Tạo vocabulary cho debug"""
    token_to_id = {
        PAD: 0, UNK: 1, 'VAR1': 2, 'VAR2': 3, 'FUN1': 4,
        'if': 5, 'else': 6, '=': 7, '+': 8, '(': 9, ')': 10, 'return': 11
    }
    return Vocabulary(token_to_id=token_to_id)


def create_debug_graph_data():
    """Tạo graph data cho debug"""
    # Create multiple small graphs
    graphs = []
    
    # Graph 1: 4 nodes
    x1 = torch.tensor([
        [2, 7, 4, 0, 0, 0],  # VAR1 = FUN1 PAD PAD PAD
        [5, 2, 0, 0, 0, 0],  # if VAR1 PAD PAD PAD PAD
        [11, 2, 0, 0, 0, 0], # return VAR1 PAD PAD PAD PAD
        [4, 9, 10, 0, 0, 0]  # FUN1 ( ) PAD PAD PAD
    ], dtype=torch.long)
    
    edge_index1 = torch.tensor([
        [0, 1, 1, 2],  # source nodes
        [1, 2, 3, 3]   # target nodes
    ], dtype=torch.long)
    
    graph1 = Data(x=x1, edge_index=edge_index1)
    graphs.append(graph1)
    
    # Graph 2: 3 nodes
    x2 = torch.tensor([
        [2, 8, 3, 0, 0, 0],  # VAR1 + VAR2 PAD PAD PAD
        [5, 2, 6, 0, 0, 0],  # if VAR1 else PAD PAD PAD
        [11, 0, 0, 0, 0, 0]  # return PAD PAD PAD PAD PAD
    ], dtype=torch.long)
    
    edge_index2 = torch.tensor([
        [0, 1],  # source nodes
        [1, 2]   # target nodes  
    ], dtype=torch.long)
    
    graph2 = Data(x=x2, edge_index=edge_index2)
    graphs.append(graph2)
    
    # Graph 3: 5 nodes
    x3 = torch.tensor([
        [2, 7, 3, 0, 0, 0],  # VAR1 = VAR2 PAD PAD PAD
        [4, 9, 2, 10, 0, 0], # FUN1 ( VAR1 ) PAD PAD
        [5, 2, 0, 0, 0, 0],  # if VAR1 PAD PAD PAD PAD
        [11, 2, 8, 3, 0, 0], # return VAR1 + VAR2 PAD PAD
        [2, 7, 4, 0, 0, 0]   # VAR1 = FUN1 PAD PAD PAD
    ], dtype=torch.long)
    
    edge_index3 = torch.tensor([
        [0, 0, 1, 2, 3, 4],  # source nodes
        [1, 4, 2, 3, 4, 0]   # target nodes
    ], dtype=torch.long)
    
    graph3 = Data(x=x3, edge_index=edge_index3)
    graphs.append(graph3)
    
    # Create batch
    batch = Batch.from_data_list(graphs)
    return batch

class DebugGraphConvEncoder(torch.nn.Module):
    """GraphConvEncoder với debug prints chi tiết"""

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int, pad_idx: int):
        super(DebugGraphConvEncoder, self).__init__()
        print(f"🔧 Initializing GraphConvEncoder...")
        print(f"   rnn_hidden_size: {config.rnn.hidden_size}")
        print(f"   gnn_hidden_size: {config.hidden_size}")
        print(f"   n_hidden_layers: {config.n_hidden_layers}")
        print(f"   pooling_ratio: {config.pooling_ratio}")
        
        self.__config = config
        self.__pad_idx = pad_idx

        # Statement encoder (already debugged separately)
        print(f"\n📝 Creating Statement Encoder...")
        from src.models.modules.common_layers import STEncoder
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        # Input GCN Layer
        print(f"\n🕸️ Creating Input GCN Layer...")
        self.input_GCL = GCNConv(config.rnn.hidden_size, config.hidden_size)
        print(f"   Input GCN: {config.rnn.hidden_size} → {config.hidden_size}")

        # Input Pooling Layer
        print(f"\n🏊 Creating Input Pooling Layer...")
        self.input_GPL = TopKPooling(config.hidden_size, ratio=config.pooling_ratio)
        print(f"   Pooling ratio: {config.pooling_ratio}")

        # Hidden layers
        print(f"\n🔗 Creating Hidden Layers...")
        for i in range(config.n_hidden_layers - 1):
            gcl_name = f"hidden_GCL{i}"
            gpl_name = f"hidden_GPL{i}"
            
            setattr(self, gcl_name, GCNConv(config.hidden_size, config.hidden_size))
            setattr(self, gpl_name, TopKPooling(config.hidden_size, ratio=config.pooling_ratio))
            print(f"   {gcl_name}: {config.hidden_size} → {config.hidden_size}")
            print(f"   {gpl_name}: ratio={config.pooling_ratio}")
        
        # Global attention pooling
        print(f"\n🎯 Creating Global Attention Pooling...")
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))
        print(f"   Attention gate: {config.hidden_size} → 1")
        
        print(f"✅ GraphConvEncoder initialization complete!\n")


    def forward(self, batched_graph: Batch):
        print(f"\n🚀 GraphConvEncoder Forward Pass")
        print(f"   Input type: {type(batched_graph)}")
        print(f"   Batch info:")
        print(f"     - x shape: {batched_graph.x.shape}")
        print(f"     - edge_index shape: {batched_graph.edge_index.shape}")
        print(f"     - batch shape: {batched_graph.batch.shape}")
        print(f"     - num_graphs: {batched_graph.num_graphs}")

        # Step 1: Statement Embedding
        print(f"\n📝 Step 1: Statement Embedding")
        node_embedding = self.__st_embedding(batched_graph.x)
        print(f"   Node embeddings shape: {node_embedding.shape}")
        print(f"   Node embeddings range: [{node_embedding.min().item():.4f}, {node_embedding.max().item():.4f}]")
        
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch

        # Step 2: Input GCN Layer
        print(f"\n🕸️ Step 2: Input GCN Layer")
        node_embedding = self._debug_gcn_layer(self.input_GCL, node_embedding, edge_index, "input_GCL")
        
        # Step 3: Input Pooling Layer
        print(f"\n🏊 Step 3: Input Pooling Layer")
        node_embedding, edge_index, _, batch, perm, score = self._debug_pooling_layer(
            self.input_GPL, node_embedding, edge_index, batch, "input_GPL"
        )
        
        # Step 4: First attention pooling
        print(f"\n🎯 Step 4: Global Attention Pooling (Layer 0)")
        out = self._debug_global_attention(self.attpool, node_embedding, batch, "layer_0")
        
        # Step 5: Hidden layers
        for i in range(self.__config.n_hidden_layers - 1):
            print(f"\n🔗 Step {5+i*2}: Hidden GCN Layer {i}")
            gcl = getattr(self, f"hidden_GCL{i}")
            node_embedding = self._debug_gcn_layer(gcl, node_embedding, edge_index, f"hidden_GCL{i}")
            
            print(f"\n🏊 Step {6+i*2}: Hidden Pooling Layer {i}")
            gpl = getattr(self, f"hidden_GPL{i}")
            node_embedding, edge_index, _, batch, perm, score = self._debug_pooling_layer(
                gpl, node_embedding, edge_index, batch, f"hidden_GPL{i}"
            )
            
            print(f"\n🎯 Global Attention Pooling (Layer {i+1})")
            layer_out = self._debug_global_attention(self.attpool, node_embedding, batch, f"layer_{i+1}")
            out += layer_out  # Residual connection
            print(f"   Residual connection: out shape after += {out.shape}")
        
        print(f"\n✅ GraphConvEncoder Forward Complete!")
        print(f"   Final output shape: {out.shape}")
        print(f"   Final output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        return out
    
    def _debug_gcn_layer(self, gcn_layer, x, edge_index, layer_name):
        """Debug GCN layer forward pass"""
        print(f"      🕸️ {layer_name} Details:")
        print(f"         Input shape: {x.shape}")
        print(f"         Edge index shape: {edge_index.shape}")
        print(f"         Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # GCN forward
        x_out = F.relu(gcn_layer(x, edge_index))
        
        print(f"         Output shape: {x_out.shape}")
        print(f"         Output range: [{x_out.min().item():.4f}, {x_out.max().item():.4f}]")
        print(f"         ReLU zeros: {(x_out == 0).sum().item()}/{x_out.numel()} ({100*(x_out == 0).sum().item()/x_out.numel():.1f}%)")
        
        return x_out
    
    def _debug_pooling_layer(self, pool_layer, x, edge_index, batch, layer_name):
        """Debug TopK pooling layer"""
        print(f"      🏊 {layer_name} Details:")
        print(f"         Input nodes: {x.shape[0]}")
        print(f"         Input edges: {edge_index.shape[1]}")
        print(f"         Input batch unique: {torch.unique(batch).tolist()}")
        print(f"         Pooling ratio: {pool_layer.ratio}")
        
        # Count nodes per graph before pooling
        unique_batch, counts_before = torch.unique(batch, return_counts=True)
        print(f"         Nodes per graph before: {counts_before.tolist()}")
        
        # TopK pooling forward
        x_out, edge_index_out, edge_attr_out, batch_out, perm, score = pool_layer(
            x, edge_index, None, batch
        )
        
        print(f"         Output nodes: {x_out.shape[0]} (kept {x_out.shape[0]/x.shape[0]*100:.1f}%)")
        print(f"         Output edges: {edge_index_out.shape[1] if edge_index_out is not None else 0}")
        print(f"         Output batch unique: {torch.unique(batch_out).tolist()}")
        
        # Count nodes per graph after pooling
        unique_batch_out, counts_after = torch.unique(batch_out, return_counts=True)
        print(f"         Nodes per graph after: {counts_after.tolist()}")
        
        # Pooling score analysis
        print(f"         Pooling scores:")
        print(f"           Min: {score.min().item():.4f}")
        print(f"           Max: {score.max().item():.4f}")
        print(f"           Mean: {score.mean().item():.4f}")
        print(f"           Std: {score.std().item():.4f}")
        
        # Selected nodes analysis
        print(f"         Selected nodes (perm): {perm[:10].tolist()}...")
        
        return x_out, edge_index_out, edge_attr_out, batch_out, perm, score
    
    def _debug_global_attention(self, attention_layer, x, batch, layer_name):
        """Debug global attention pooling"""
        print(f"      🎯 Global Attention ({layer_name}):")
        print(f"         Input shape: {x.shape}")
        print(f"         Batch shape: {batch.shape}")
        print(f"         Num graphs: {torch.unique(batch).shape[0]}")
        
        # Attention weights computation
        gate_linear = attention_layer.gate_nn
        attention_weights = gate_linear(x)  # [num_nodes, 1]
        print(f"         Attention weights shape: {attention_weights.shape}")
        print(f"         Attention weights range: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
        
        # Global attention forward
        out = attention_layer(x, batch)
        
        print(f"         Output shape: {out.shape}")
        print(f"         Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        return out

def debug_gcn_encoder():
    """Debug GraphConvEncoder"""
    print("🐛 Starting GraphConvEncoder Debug Session")
    print("=" * 60)
    
    # Setup
    config = create_debug_config()
    vocab = create_debug_vocab()
    batch_data = create_debug_graph_data()
    
    print(f"📊 Debug Setup:")
    print(f"   Vocab size: {vocab.get_vocab_size()}")
    print(f"   Batch info: {batch_data}")
    print(f"   Config: {config}")
    
    vocab_size = vocab.get_vocab_size() 
    # Initialize model
    print(f"\n🏗️ Model Initialization:")
    model = DebugGraphConvEncoder(config, vocab, vocab_size, vocab.get_pad_id())
    
    # Forward pass
    print(f"\n🚀 Running Forward Pass:")
    print("=" * 60)
    
    with torch.no_grad():
        output = model(batch_data)
    
    print(f"\n✅ Debug Complete!")
    print(f"   Final output shape: {output.shape}")
    print(f"   Expected: [num_graphs={batch_data.num_graphs}, hidden_size={config.hidden_size}]")
    print(f"   Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    return model, output

def main():
    """Main debug function"""
    print("🎯 GNN Layers Debug Menu")
    print("=" * 40)
    print("1. Debug GraphConvEncoder")
    
        
    debug_gcn_encoder()
 


if __name__ == "__main__":
    main()