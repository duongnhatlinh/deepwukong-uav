#!/usr/bin/env python3
"""
Debug script để trace qua hidden layers và classifier của DeepWukong
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from omegaconf import DictConfig
import torch.nn.functional as F

class DebugHiddenClassifier(nn.Module):
    """Hidden layers và Classifier với debug prints chi tiết"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        print(f"🔧 Initializing Hidden Layers and Classifier...")
        print(f"   gnn.hidden_size: {config.gnn.hidden_size}")
        print(f"   classifier.hidden_size: {config.classifier.hidden_size}")
        print(f"   classifier.n_hidden_layers: {config.classifier.n_hidden_layers}")
        print(f"   classifier.drop_out: {config.classifier.drop_out}")
        print(f"   classifier.n_classes: {config.classifier.n_classes}")
        
        self.__config = config
        hidden_size = config.classifier.hidden_size
        
        # Build hidden layers
        print(f"\n🔗 Building Hidden Layers...")
        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        print(f"   Layer 0: Linear({config.gnn.hidden_size} → {hidden_size})")
        print(f"   Layer 0: ReLU()")
        print(f"   Layer 0: Dropout(p={config.classifier.drop_out})")
        
        for i in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
            print(f"   Layer {i+1}: Linear({hidden_size} → {hidden_size})")
            print(f"   Layer {i+1}: ReLU()")
            print(f"   Layer {i+1}: Dropout(p={config.classifier.drop_out})")
        
        self.__hidden_layers = nn.Sequential(*layers)
        
        # Build classifier
        print(f"\n🎯 Building Classifier...")
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)
        print(f"   Classifier: Linear({hidden_size} → {config.classifier.n_classes})")
        
        print(f"✅ Hidden Layers and Classifier initialization complete!\n")
    
    def forward(self, graph_embeddings: torch.Tensor):
        print(f"\n🚀 Hidden Layers and Classifier Forward Pass")
        print(f"   Input shape: {graph_embeddings.shape}")
        print(f"   Input stats: min={graph_embeddings.min().item():.4f}, "
              f"max={graph_embeddings.max().item():.4f}, "
              f"mean={graph_embeddings.mean().item():.4f}")
        
        # Trace through hidden layers
        print(f"\n🔗 Processing Hidden Layers...")
        x = graph_embeddings
        
        layer_idx = 0
        for module in self.__hidden_layers:
            old_shape = x.shape
            x = module(x)
            
            if isinstance(module, nn.Linear):
                print(f"\n   Layer {layer_idx} - Linear:")
                print(f"      Input shape: {old_shape}")
                print(f"      Output shape: {x.shape}")
                print(f"      Weight shape: {module.weight.shape}")
                print(f"      Bias shape: {module.bias.shape}")
                print(f"      Output stats: min={x.min().item():.4f}, "
                      f"max={x.max().item():.4f}, "
                      f"mean={x.mean().item():.4f}")
                
            elif isinstance(module, nn.ReLU):
                print(f"   Layer {layer_idx} - ReLU:")
                print(f"      Negative values before: {(old_shape[0] * old_shape[1]) - (x > 0).sum().item()}")
                print(f"      Positive values after: {(x > 0).sum().item()}")
                print(f"      Zero values: {(x == 0).sum().item()}")
                
            elif isinstance(module, nn.Dropout):
                print(f"   Layer {layer_idx} - Dropout:")
                if self.training:
                    print(f"      Dropout rate: {module.p}")
                    print(f"      Expected zeros: ~{int(x.numel() * module.p)}")
                    print(f"      Actual zeros: {(x == 0).sum().item()}")
                else:
                    print(f"      Dropout inactive (eval mode)")
                layer_idx += 1
        
        hiddens = x
        print(f"\n   Hidden layers output shape: {hiddens.shape}")
        print(f"   Hidden layers output stats: min={hiddens.min().item():.4f}, "
              f"max={hiddens.max().item():.4f}, "
              f"mean={hiddens.mean().item():.4f}")
        
        # Classifier
        print(f"\n🎯 Processing Classifier...")
        logits = self.__classifier(hiddens)
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits stats: min={logits.min().item():.4f}, "
              f"max={logits.max().item():.4f}, "
              f"mean={logits.mean().item():.4f}")
        
        # Analyze predictions
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            _, preds = logits.max(dim=1)
            
            print(f"\n   Predictions analysis:")
            print(f"      Predicted class 0 (safe): {(preds == 0).sum().item()}")
            print(f"      Predicted class 1 (vulnerable): {(preds == 1).sum().item()}")
            print(f"      Average confidence: {probs.max(dim=1)[0].mean().item():.4f}")
            
            # Show sample predictions
            print(f"\n   Sample predictions (first 5):")
            for i in range(min(5, logits.shape[0])):
                print(f"      Graph {i}: logits={logits[i].tolist()}, "
                      f"probs={probs[i].tolist()}, "
                      f"pred={preds[i].item()}")
        
        return logits


def create_mock_graph_embeddings(batch_size: int, hidden_size: int):
    """Create mock graph embeddings to test hidden layers"""
    # Simulate realistic graph embeddings with some patterns
    embeddings = torch.randn(batch_size, hidden_size)
    
    # Add some structure - make some graphs more "vulnerable-like"
    vulnerable_indices = torch.randint(0, batch_size, (batch_size // 3,))
    embeddings[vulnerable_indices] += torch.randn(len(vulnerable_indices), hidden_size) * 0.5
    
    return embeddings


def create_debug_config():
    """Create config for debugging"""
    return DictConfig({
        'gnn': {
            'hidden_size': 256
        },
        'classifier': {
            'hidden_size': 512,
            'n_hidden_layers': 2,
            'n_classes': 2,
            'drop_out': 0.5
        }
    })


def main():
    """Main debug function"""
    print("🐛 Starting Hidden Layers and Classifier Debug Session")
    print("=" * 60)
    
    # Setup
    config = create_debug_config()
    batch_size = 8
    
    print(f"📊 Debug Setup:")
    print(f"   Batch size: {batch_size}")
    print(f"   Config: {config}")
    
    # Initialize model
    print(f"\n🏗️ Model Initialization:")
    model = DebugHiddenClassifier(config)
    
    # Create mock input
    print(f"\n📥 Creating Mock Input:")
    graph_embeddings = create_mock_graph_embeddings(batch_size, config.gnn.hidden_size)
    print(f"   Mock embeddings shape: {graph_embeddings.shape}")
    
    # Test in training mode
    print(f"\n🚀 Running Forward Pass (Training Mode):")
    print("=" * 60)
    model.train()
    with torch.no_grad():  # No gradients needed for debugging
        logits_train = model(graph_embeddings)
    
    # Test in eval mode
    print(f"\n🚀 Running Forward Pass (Eval Mode):")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        logits_eval = model(graph_embeddings)
    
    # Compare outputs
    print(f"\n📊 Comparing Training vs Eval Mode:")
    print(f"   Logits difference: {(logits_train - logits_eval).abs().max().item():.6f}")
    print(f"   Same predictions: {(logits_train.argmax(dim=1) == logits_eval.argmax(dim=1)).sum().item()}/{batch_size}")
    
    # Test with edge cases
    print(f"\n🔍 Testing Edge Cases:")
    
    # All zeros
    print(f"\n   Test 1: All zeros input")
    zero_input = torch.zeros(2, config.gnn.hidden_size)
    model.eval()
    with torch.no_grad():
        zero_output = model(zero_input)
    print(f"   Zero input → Output: {zero_output}")
    
    # Very large values
    print(f"\n   Test 2: Large values input")
    large_input = torch.randn(2, config.gnn.hidden_size) * 100
    with torch.no_grad():
        large_output = model(large_input)
    print(f"   Large input stats: min={large_input.min():.2f}, max={large_input.max():.2f}")
    print(f"   Large input → Output: {large_output}")
    
    print(f"\n✅ Debug Complete!")
    
    return model, logits_eval


if __name__ == "__main__":
    model, output = main()