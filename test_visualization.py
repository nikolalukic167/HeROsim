#!/usr/bin/env python3
"""
Test file for network topology visualization.
You can modify the configuration and test different scenarios.
"""

import sys
import os
import json
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from executeinitial import prepare_simulation_config, visualize_network_topology

def test_visualization():
    """Test the network topology visualization."""
    print("=== Network Topology Visualization Test ===")
    
    # Load configuration
    config_path = "simulation_data/space_with_network.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Client nodes: {config['nodes']['client_nodes']['count']}")
    print(f"Server nodes: {config['nodes']['server_nodes']['count']}")
    
    # Create a mock sample
    sample_array = np.array([100])  # network_bandwidth
    mock_mapping = {0: 'network_bandwidth'}
    
    try:
        # Generate infrastructure with visualization
        print("\nGenerating infrastructure with visualization...")
        infrastructure = prepare_simulation_config(sample_array, mock_mapping, config)
        
        # Check if visualization was created
        viz_path = "simulation_data/network_topology.pdf"
        if os.path.exists(viz_path):
            print(f"✓ Visualization created: {viz_path}")
            print(f"  File size: {os.path.getsize(viz_path)} bytes")
            
            # Show some statistics
            total_nodes = len(infrastructure['nodes'])
            client_nodes = [n for n in infrastructure['nodes'] if n['node_name'].startswith('client_node')]
            server_nodes = [n for n in infrastructure['nodes'] if not n['node_name'].startswith('client_node')]
            
            print(f"\nInfrastructure Statistics:")
            print(f"  Total nodes: {total_nodes}")
            print(f"  Client nodes: {len(client_nodes)}")
            print(f"  Server nodes: {len(server_nodes)}")
            
            # Count connections
            total_connections = sum(len(node['network_map']) for node in infrastructure['nodes'])
            print(f"  Total connections: {total_connections}")
            print(f"  Average connections per node: {total_connections / total_nodes:.1f}")
            
        else:
            print("✗ Visualization file was not created")
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_custom_config():
    """Test with custom node counts."""
    print("\n=== Testing Custom Configuration ===")
    
    # Create a custom configuration with different node counts
    custom_config = {
        "nodes": {
            "client_nodes": {"count": 10},  # Fewer clients
            "server_nodes": {"count": 5}    # Fewer servers
        },
        "pci": {
            "rpi": {
                "specs": {
                    "node_name": "",
                    "network_map": {},
                    "memory": 8,
                    "platforms": ["rpiCpu"],
                    "storage": ["flashCard"]
                }
            }
        },
        "network": {
            "latency": {
                "base_latency": 0.1,
                "device_latencies": {
                    "rpi": {"rpi": {"min": 0.05, "max": 0.15}}
                }
            }
        }
    }
    
    # Create mock sample and mapping
    sample_array = np.array([100])
    mock_mapping = {0: 'network_bandwidth'}
    
    try:
        print("Testing with 10 clients and 5 servers...")
        infrastructure = prepare_simulation_config(sample_array, mock_mapping, custom_config)
        
        # Create custom visualization
        custom_viz_path = "simulation_data/network_topology_custom.pdf"
        
        # Extract network maps for visualization
        network_maps = {node['node_name']: node['network_map'] for node in infrastructure['nodes']}
        visualize_network_topology(network_maps, custom_viz_path)
        
        if os.path.exists(custom_viz_path):
            print(f"✓ Custom visualization created: {custom_viz_path}")
            print(f"  File size: {os.path.getsize(custom_viz_path)} bytes")
        else:
            print("✗ Custom visualization file was not created")
            
    except Exception as e:
        print(f"✗ Custom test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_visualization_only():
    """Test just the visualization function with sample data."""
    print("\n=== Testing Visualization Function Only ===")
    
    # Create sample network maps
    sample_network_maps = {}
    
    # Add some client nodes
    for i in range(5):
        client_name = f"client_node{i}"
        sample_network_maps[client_name] = {}
    
    # Add some server nodes
    for i in range(3):
        server_name = f"node{i}"
        sample_network_maps[server_name] = {}
    
    # Add some connections
    for i in range(5):
        client_name = f"client_node{i}"
        for j in range(3):
            server_name = f"node{j}"
            latency = 0.1 + (i + j) * 0.01
            sample_network_maps[client_name][server_name] = latency
            sample_network_maps[server_name][client_name] = latency
    
    try:
        test_viz_path = "simulation_data/network_topology_test.pdf"
        visualize_network_topology(sample_network_maps, test_viz_path)
        
        if os.path.exists(test_viz_path):
            print(f"✓ Test visualization created: {test_viz_path}")
            print(f"  File size: {os.path.getsize(test_viz_path)} bytes")
        else:
            print("✗ Test visualization file was not created")
            
    except Exception as e:
        print(f"✗ Visualization test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Network Topology Visualization Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_visualization()
    test_custom_config()
    test_visualization_only()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nGenerated files:")
    print("- simulation_data/network_topology.pdf (main visualization)")
    print("- simulation_data/network_topology_custom.pdf (custom config)")
    print("- simulation_data/network_topology_test.pdf (test visualization)") 