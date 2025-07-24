#!/usr/bin/env python3
"""
WRR GNN Model Architecture Visualization
This script creates comprehensive visualizations of the WRR GNN model architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import os
import sys

# Add the project root to the path to import the model
sys.path.append('/root/projects/my-herosim')

# Import the model configuration and architecture
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the WRR-based GNN model"""
    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    attention_heads: int = 4
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    patience: int = 15
    min_delta: float = 1e-4
    
    # WRR-specific configuration
    use_composite_score: bool = True
    score_normalization: str = 'sigmoid'
    
    # Composite score weights
    weight_latency: float = 0.4
    weight_queue: float = 0.25
    weight_cold_start: float = 0.15
    weight_energy: float = 0.1
    weight_utilization: float = 0.1
    
    # Data processing
    normalize_features: bool = True
    num_physical_nodes: int = 10

# Import PyTorch Geometric components
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv

class WRRNodeBasedGNN(torch.nn.Module):
    """Node-based GNN that predicts WRR composite scores for physical nodes"""
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, config: ModelConfig):
        super(WRRNodeBasedGNN, self).__init__()

        self.config = config
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # Input encoders
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(node_feature_dim, config.hidden_dim),
            torch.nn.BatchNorm1d(config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout)
        )
        
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_dim, config.hidden_dim),
            torch.nn.BatchNorm1d(config.hidden_dim),
            torch.nn.ReLU()
        )
        
        # Graph convolution layers
        self.conv_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for i in range(config.num_layers):
            self.conv_layers.append(
                GATConv(
                    config.hidden_dim, 
                    config.hidden_dim // config.attention_heads,
                    heads=config.attention_heads, 
                    dropout=config.dropout,
                    edge_dim=config.hidden_dim
                )
            )
            self.batch_norms.append(BatchNorm(config.hidden_dim))
        
        # Output layers - predict WRR composite scores
        self.wrr_score_predictor = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim // 2, 1),
            torch.nn.Sigmoid() if config.score_normalization == 'sigmoid' else torch.nn.Identity()
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Encode inputs
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply graph convolutions with residual connections
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.config.dropout, training=self.training)
            
            # Residual connection
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Predict WRR composite scores
        wrr_score_pred = self.wrr_score_predictor(x)
        
        return {
            'wrr_score': wrr_score_pred.squeeze(-1)
        }

def visualize_wrr_gnn_architecture(config, model):
    """Create a comprehensive visualization of the WRR GNN model architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors for different components
    colors = {
        'input': '#E8F4FD',
        'encoder': '#B3D9FF', 
        'conv': '#FFB366',
        'attention': '#FF9999',
        'output': '#90EE90',
        'residual': '#DDA0DD',
        'text': '#2E2E2E'
    }
    
    # Title
    ax.text(5, 11.5, 'WRR GNN Model Architecture', fontsize=20, fontweight='bold', 
            ha='center', color=colors['text'])
    ax.text(5, 11.2, 'Weighted Round-Robin Graph Neural Network for Task Scheduling', 
            fontsize=14, ha='center', color='gray')
    
    # Input Layer
    input_box = FancyBboxPatch((0.5, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10, 'Input Features\n(24 dim/node)', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=colors['text'])
    
    # Node Encoder
    encoder_box = FancyBboxPatch((3.5, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                                facecolor=colors['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(4.5, 10, 'Node Encoder\nLinear + BN + ReLU', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=colors['text'])
    
    # Edge Encoder
    edge_encoder_box = FancyBboxPatch((6.5, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                                     facecolor=colors['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(edge_encoder_box)
    ax.text(7.5, 10, 'Edge Encoder\nLinear + BN + ReLU', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=colors['text'])
    
    # Graph Convolution Layers
    conv_layers = []
    for i in range(4):
        y_pos = 8 - i * 1.5
        conv_box = FancyBboxPatch((2, y_pos), 6, 1, boxstyle="round,pad=0.1", 
                                 facecolor=colors['conv'], edgecolor='black', linewidth=2)
        ax.add_patch(conv_box)
        
        layer_text = f'GAT Layer {i+1}\n{config.attention_heads} heads, {config.hidden_dim} dim'
        ax.text(5, y_pos + 0.5, layer_text, fontsize=10, ha='center', va='center', 
                fontweight='bold', color=colors['text'])
        
        conv_layers.append((5, y_pos + 0.5))
    
    # Residual Connections
    for i in range(3):
        start_y = 8 - i * 1.5
        end_y = 6.5 - i * 1.5
        ax.arrow(1, start_y + 0.5, 0, end_y - start_y, head_width=0.1, head_length=0.1, 
                fc=colors['residual'], ec=colors['residual'], linewidth=3, alpha=0.7)
        ax.text(0.5, (start_y + end_y) / 2 + 0.5, 'Residual\nConnection', fontsize=8, 
                ha='center', va='center', color=colors['residual'], fontweight='bold')
    
    # Attention Mechanism Detail
    attention_box = FancyBboxPatch((0.5, 1.5), 2, 1, boxstyle="round,pad=0.1", 
                                  facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(1.5, 2, 'Multi-Head\nAttention', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=colors['text'])
    
    # Output Layer
    output_box = FancyBboxPatch((3.5, 1.5), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(4.5, 2, 'WRR Score\nPredictor', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=colors['text'])
    
    # Final Output
    final_box = FancyBboxPatch((6.5, 1.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(7.5, 2, 'Composite\nWRR Score', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=colors['text'])
    
    # Arrows connecting components
    arrows = [
        # Input to encoders
        ((2.5, 10), (3.5, 10)),  # Input to Node Encoder
        ((2.5, 10), (6.5, 10)),  # Input to Edge Encoder
        
        # Encoders to first conv layer
        ((4.5, 9.5), (5, 8.5)),  # Node Encoder to Conv1
        ((7.5, 9.5), (5, 8.5)),  # Edge Encoder to Conv1
        
        # Between conv layers
        ((5, 7.5), (5, 7)),      # Conv1 to Conv2
        ((5, 6), (5, 5.5)),      # Conv2 to Conv3
        ((5, 4.5), (5, 4)),      # Conv3 to Conv4
        
        # Conv to attention
        ((5, 2.5), (1.5, 2.5)),  # Conv4 to Attention
        
        # Attention to output
        ((2.5, 2), (3.5, 2)),    # Attention to WRR Predictor
        
        # Output to final
        ((5.5, 2), (6.5, 2)),    # WRR Predictor to Final Score
    ]
    
    for start, end in arrows:
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
                head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Model specifications
    specs_text = f"""
Model Specifications:
• Hidden Dimension: {config.hidden_dim}
• Number of Layers: {config.num_layers}
• Attention Heads: {config.attention_heads}
• Dropout Rate: {config.dropout}
• Total Parameters: {sum(p.numel() for p in model.parameters()):,}
• Input Features: 24 per node
• Edge Features: 2 per edge
• Output: WRR Composite Score (0-1)
    """
    
    ax.text(8.5, 8, specs_text, fontsize=10, va='top', color=colors['text'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # WRR Score Components
    wrr_text = f"""
WRR Score Components:
• Latency Weight: {config.weight_latency}
• Queue Weight: {config.weight_queue}
• Cold Start Weight: {config.weight_cold_start}
• Energy Weight: {config.weight_energy}
• Utilization Weight: {config.weight_utilization}
    """
    
    ax.text(8.5, 4, wrr_text, fontsize=10, va='top', color=colors['text'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualize_model_layers(config, model):
    """Create a layer-by-layer visualization of the model"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Layer information
    layers = [
        ('Input', 24, '#E8F4FD'),
        ('Node Encoder', config.hidden_dim, '#B3D9FF'),
        ('Edge Encoder', config.hidden_dim, '#B3D9FF'),
        ('GAT Layer 1', config.hidden_dim, '#FFB366'),
        ('GAT Layer 2', config.hidden_dim, '#FFB366'),
        ('GAT Layer 3', config.hidden_dim, '#FFB366'),
        ('GAT Layer 4', config.hidden_dim, '#FFB366'),
        ('WRR Predictor', config.hidden_dim // 2, '#90EE90'),
        ('Output', 1, '#90EE90')
    ]
    
    y_positions = np.linspace(1, 9, len(layers))
    
    for i, (name, dim, color) in enumerate(layers):
        # Layer box
        box = FancyBboxPatch((1, y_positions[i] - 0.3), 3, 0.6, 
                            boxstyle="round,pad=0.1", facecolor=color, 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Layer name and dimension
        ax.text(2.5, y_positions[i], f'{name}\n{dim} dim', fontsize=10, 
                ha='center', va='center', fontweight='bold')
        
        # Arrow to next layer
        if i < len(layers) - 1:
            ax.arrow(2.5, y_positions[i] - 0.3, 0, y_positions[i+1] - y_positions[i] - 0.3,
                    head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Title
    ax.text(5, 9.5, 'WRR GNN Layer Architecture', fontsize=16, fontweight='bold', ha='center')
    
    # Model stats
    stats_text = f"""
Model Statistics:
• Total Parameters: {sum(p.numel() for p in model.parameters()):,}
• Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
• Model Size: {sum(p.numel() for p in model.parameters()) * 4 / 1024:.1f} KB
• Architecture: GAT with Residual Connections
• Activation: ReLU + Dropout
• Output: Sigmoid-normalized WRR Score
    """
    
    ax.text(6, 7, stats_text, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create and display the visualizations"""
    
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize model
    model = WRRNodeBasedGNN(
        node_feature_dim=24,
        edge_feature_dim=2,
        config=config
    )
    
    print("🎨 Creating WRR GNN Model Visualizations...")
    
    # Create comprehensive architecture visualization
    fig1 = visualize_wrr_gnn_architecture(config, model)
    plt.figure(fig1.number)
    plt.savefig('/root/projects/my-herosim/src/notebooks/wrr_gnn_architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create layer-by-layer visualization
    fig2 = visualize_model_layers(config, model)
    plt.figure(fig2.number)
    plt.savefig('/root/projects/my-herosim/src/notebooks/wrr_gnn_layers.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model architecture visualizations created successfully!")
    print(f"📊 Model has {sum(p.numel() for p in model.parameters()):,} total parameters")
    print(f"🎯 Output: WRR Composite Score (0-1 range)")
    print(f"🏗️  Architecture: {config.num_layers}-layer GAT with {config.attention_heads} attention heads")
    print(f"💾 Visualizations saved to:")
    print(f"   - /root/projects/my-herosim/src/notebooks/wrr_gnn_architecture.png")
    print(f"   - /root/projects/my-herosim/src/notebooks/wrr_gnn_layers.png")

if __name__ == "__main__":
    main() 