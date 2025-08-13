import torch
import torch.nn as nn
from omegaconf import DictConfig
import numpy as np
from src.vocabulary import Vocabulary
from src.utils import PAD, UNK


def create_debug_config():
    """Tạo config cho debug"""
    return DictConfig({
        'embed_size': 128,  # Giảm để debug dễ hơn
        'rnn': {
            'hidden_size': 64,
            'num_layers': 1,
            'use_bi': True,
            'drop_out': 0.1
        }
    })

def create_debug_vocab():
    """Tạo vocabulary nhỏ cho debug"""
    token_to_id = {
        PAD: 0,
        UNK: 1,
        'VAR1': 2,
        'VAR2': 3,
        'FUN1': 4,
        'if': 5,
        'else': 6,
        '=': 7,
        '+': 8,
        '(': 9,
        ')': 10,
        'return': 11
    }
    return Vocabulary(token_to_id=token_to_id)

def create_debug_data():
    """Tạo dữ liệu test nhỏ"""
    # 5 nodes, mỗi node có tối đa 8 tokens
    sequences = [
        [2, 7, 4, 9, 10, 0, 0, 0],  # VAR1 = FUN1 ( ) PAD PAD PAD (length=5)
        [5, 2, 0, 0, 0, 0, 0, 0],   # if VAR1 PAD PAD PAD PAD PAD PAD (length=2)
        [11, 2, 8, 3, 0, 0, 0, 0],  # return VAR1 + VAR2 PAD PAD PAD PAD (length=4)
        [2, 7, 3, 0, 0, 0, 0, 0],   # VAR1 = VAR2 PAD PAD PAD PAD PAD (length=3)
        [4, 9, 2, 10, 0, 0, 0, 0]   # FUN1 ( VAR1 ) PAD PAD PAD PAD (length=4)
    ]
    return torch.tensor(sequences, dtype=torch.long)

class DebugRNNLayer(torch.nn.Module):
    """RNN Layer với debug prints chi tiết"""
    
    def __init__(self, config: DictConfig, pad_idx: int):
        super(DebugRNNLayer, self).__init__()
        print(f"   🔄 Initializing RNN Layer...")
        
        self.__pad_idx = pad_idx
        self.__config = config

        # LSTM parameters
        print(f"      input_size: {config.embed_size}")
        print(f"      hidden_size: {config.rnn.hidden_size}")
        print(f"      num_layers: {config.rnn.num_layers}")
        print(f"      bidirectional: {config.rnn.use_bi}")
        print(f"      dropout: {config.rnn.drop_out}")

        self.__rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.rnn.hidden_size,
            num_layers=config.rnn.num_layers,
            bidirectional=config.rnn.use_bi,
            dropout=config.rnn.drop_out if config.rnn.num_layers > 1 else 0,
            batch_first=True
        )

        self.__dropout_rnn = nn.Dropout(config.rnn.drop_out)
        print(f"   ✅ RNN Layer initialized")

    def forward(self, subtokens_embed: torch.Tensor, node_ids: torch.Tensor):
        print(f"\n      🔄 RNN Forward Pass")
        print(f"         Input embeddings shape: {subtokens_embed.shape}")
        print(f"         Input node_ids shape: {node_ids.shape}")

        # Step 1: Calculate sequence lengths
        print(f"\n      📏 Step 1: Calculate sequence lengths")
        with torch.no_grad():
            # Tìm vị trí PAD đầu tiên trong mỗi sequence
            is_contain_pad_id, first_pad_pos = torch.max(node_ids == self.__pad_idx, dim=1)
            print(f"         is_contain_pad_id shape: {is_contain_pad_id.shape}")
            print(f"         first_pad_pos shape: {first_pad_pos.shape}")
            print(f"         Sample first_pad_pos (first 10): {first_pad_pos[:10]}")
            
            # Nếu không có PAD, length = max_length
            first_pad_pos[~is_contain_pad_id] = node_ids.shape[1]
            print(f"         Adjusted lengths (first 10): {first_pad_pos[:10]}")
            print(f"         Length stats: min={first_pad_pos.min()}, max={first_pad_pos.max()}, mean={first_pad_pos.float().mean():.2f}")
            
            # Sort theo length giảm dần
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            print(f"         Sorted lengths (first 10): {sorted_path_lengths[:10]}")
            print(f"         Sort indices (first 10): {sort_indices[:10]}")
            
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))

        # Step 2: Sort embeddings
        print(f"\n      🔀 Step 2: Sort embeddings by length")
        original_shape = subtokens_embed.shape
        subtokens_embed = subtokens_embed[sort_indices]
        print(f"         Before sort: {original_shape}")
        print(f"         After sort: {subtokens_embed.shape}")
        
        # Step 3: Pack sequences
        print(f"\n      📦 Step 3: Pack sequences")
        try:
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                subtokens_embed, sorted_path_lengths, batch_first=True
            )
            print(f"         Packed data shape: {packed_embeddings.data.shape}")
            print(f"         Batch sizes: {packed_embeddings.batch_sizes[:10]}")  # First 10 time steps
        except Exception as e:
            print(f"         ❌ Packing failed: {e}")
            # Fallback: process without packing
            packed_embeddings = subtokens_embed

        
        # Step 4: LSTM forward
        print(f"\n      🧠 Step 4: LSTM forward")
        try:
            lstm_output, (node_embedding, cell_state) = self.__rnn(packed_embeddings)
            print(f"         LSTM output type: {type(lstm_output)}")
            print(f"         Hidden state shape: {node_embedding.shape}")
            print(f"         Cell state shape: {cell_state.shape}")
            
            # Analyze hidden state
            print(f"         Hidden state stats:")
            print(f"            Min: {node_embedding.min().item():.4f}")
            print(f"            Max: {node_embedding.max().item():.4f}")
            print(f"            Mean: {node_embedding.mean().item():.4f}")
            print(f"            Std: {node_embedding.std().item():.4f}")
            
        except Exception as e:
            print(f"         ❌ LSTM forward failed: {e}")
            return torch.zeros(subtokens_embed.shape[0], self.__config.rnn.hidden_size)
        
        # Step 5: Combine bidirectional outputs
        print(f"\n      🔗 Step 5: Combine bidirectional outputs")
        if self.__config.rnn.use_bi:
            print(f"         Before combine: {node_embedding.shape}")
            node_embedding = node_embedding.sum(dim=0)  # Sum forward + backward
            print(f"         After combine: {node_embedding.shape}")
        else:
            node_embedding = node_embedding.squeeze(0)
        
        # Step 6: Dropout
        print(f"\n      💧 Step 6: Apply dropout")
        print(f"         Before dropout: {node_embedding.shape}")
        node_embedding = self.__dropout_rnn(node_embedding)
        print(f"         After dropout: {node_embedding.shape}")
        
        # Step 7: Restore original order
        print(f"\n      🔄 Step 7: Restore original order")
        print(f"         Before restore: {node_embedding.shape}")
        node_embedding = node_embedding[reverse_sort_indices]
        print(f"         After restore: {node_embedding.shape}")
        print(f"         Final output sample (first 3 nodes, first 5 dims):")
        print(f"         {node_embedding[:3, :5]}")
        
        return node_embedding

class DebugSTEncoder(torch.nn.Module):
    """STEncoder với debug prints chi tiết"""

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int, pad_idx: int):
        super(DebugSTEncoder, self).__init__()
        print(f"🔧 Initializing STEncoder...")
        print(f"   vocabulary_size: {vocabulary_size}")
        print(f"   embed_size: {config.embed_size}")
        print(f"   pad_idx: {pad_idx}")

        self.__config = config
        self.__pad_idx = pad_idx

        # 1. Word Embedding Layer
        print(f"\n📚 Creating Word Embedding Layer...")
        self.__wd_embedding = nn.Embedding(vocabulary_size, config.embed_size, padding_idx=pad_idx)
        print(f"   Embedding weight shape: {self.__wd_embedding.weight.shape}")

         # 2. Xavier initialization
        print(f"\n⚡ Xavier initialization...")
        torch.nn.init.xavier_uniform_(self.__wd_embedding.weight.data)
        print(f"   Weight range: [{self.__wd_embedding.weight.min().item():.4f}, {self.__wd_embedding.weight.max().item():.4f}]")

         # 3. RNN Layer
        print(f"\n🔄 Creating RNN Layer...")
        self.__rnn_attn = DebugRNNLayer(config, pad_idx)
        
        print(f"✅ STEncoder initialization complete!\n")

    def forward(self, seq: torch.Tensor):
        print(f"\n🚀 STEncoder Forward Pass")
        print(f"   Input shape: {seq.shape}")
        print(f"   Input dtype: {seq.dtype}")
        print(f"   Sample input (first 3 nodes, first 8 tokens):")
        print(f"   {seq[:3, :8]}")
        
        # Step 1: Word Embedding
        print(f"\n📚 Step 1: Word Embedding")
        print(f"   Before embedding: {seq.shape}")

        wd_embedding = self.__wd_embedding(seq)
        
        print(f"   After embedding: {wd_embedding.shape}")
        print(f"   Embedding range: [{wd_embedding.min().item():.4f}, {wd_embedding.max().item():.4f}]")
        print(f"   Sample embedding (node 0, token 0, first 5 dims): {wd_embedding[0, 0, :5]}")
        
        # Check PAD embeddings are zero
        pad_positions = (seq == self.__pad_idx)
        print(f"   PAD positions (count): {pad_positions.sum().item()}")
        print(f"   PAD positions (shape): {pad_positions.shape}")
        if pad_positions.any():
            pad_embeddings = wd_embedding[pad_positions]
            print(f"   PAD embeddings norm: {torch.norm(pad_embeddings, dim=-1).max().item():.6f} (should be ~0)")
        
        # Step 2: RNN Processing
        print(f"\n🔄 Step 2: RNN Processing")
        node_embedding = self.__rnn_attn(wd_embedding, seq)
        
        print(f"   Final output shape: {node_embedding.shape}")
        print(f"   Output range: [{node_embedding.min().item():.4f}, {node_embedding.max().item():.4f}]")
        print(f"   Sample output (first 3 nodes, first 5 dims):")
        print(f"   {node_embedding[:3, :5]}")
        
        return node_embedding
        
    
def main():
    """Main debug function"""
    print("🐛 Starting STEncoder Debug Session")
    print("=" * 60)

    # Setup
    config = create_debug_config()
    vocab = create_debug_vocab()
    test_data = create_debug_data()

    vocab_size = vocab.get_vocab_size()

    print(f"📊 Debug Setup:")
    print(f"   Vocab:{vocab}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Test data shape: {test_data.shape}")
    print(f"   Config: {config}")

    # Initialize model
    print(f"\n🏗️ Model Initialization:")
    model = DebugSTEncoder(config, vocab, vocab_size, vocab.get_pad_id())
    
    # Forward pass
    print(f"\n🚀 Running Forward Pass:")
    print("=" * 60)
    
    with torch.no_grad():  # No gradients for debugging
        output = model(test_data)
    
    print(f"\n✅ Debug Complete!")
    print(f"   Final output shape: {output.shape}")
    print(f"   Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    return model, output

if __name__ == "__main__":
    main()