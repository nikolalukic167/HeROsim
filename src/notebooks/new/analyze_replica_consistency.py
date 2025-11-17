#!/usr/bin/env python3
"""
Analyze if executecosimulation.py and simulation.py create the exact same replicas.

This script compares the replica creation logic between:
1. executecosimulation.py (lines 603-659) - used for generating placement combinations
2. simulation.py (lines 211-342) - used for actually creating replicas in simulation

Key differences to check:
1. Node iteration order
2. Platform selection logic
3. Platform assignment tracking (assigned_platforms in simulation.py)
4. Statistical distribution support (simulation.py only)
5. Platform ordering
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

def analyze_replica_creation_differences():
    """
    Analyze the differences between executecosimulation.py and simulation.py replica creation.
    """
    print("="*80)
    print("REPLICA CREATION CONSISTENCY ANALYSIS")
    print("="*80)
    print()
    
    print("KEY DIFFERENCES FOUND:")
    print()
    
    print("1. NODE ITERATION ORDER:")
    print("   executecosimulation.py (line 616):")
    print("     for node in nodes:  # Iterates through ALL nodes")
    print("       if node_name in preinit_servers:")
    print()
    print("   simulation.py (line 228):")
    print("     for node in server_nodes:  # Pre-filtered server nodes only")
    print("       if node.node_name in preinit_servers:")
    print()
    print("   ⚠️  DIFFERENCE: executecosimulation.py iterates through ALL nodes,")
    print("      while simulation.py only iterates through pre-filtered server_nodes.")
    print("      However, both check if node is in preinit_servers, so result should be same.")
    print()
    
    print("2. PLATFORM ASSIGNMENT TRACKING:")
    print("   executecosimulation.py:")
    print("     - Does NOT track assigned platforms")
    print("     - Can create multiple replicas on same platform for different task types")
    print("     - Each task type gets its own set of replicas independently")
    print()
    print("   simulation.py (line 242):")
    print("     - Uses assigned_platforms set to prevent double-booking")
    print("     - A platform can only have ONE replica across ALL task types")
    print("     - Filter: (node, platform) not in assigned_platforms")
    print()
    print("   ⚠️  CRITICAL DIFFERENCE: This is the main discrepancy!")
    print("      - executecosimulation.py: Platform can be used by multiple task types")
    print("      - simulation.py: Platform can only be used by ONE task type")
    print("      - This means simulation.py will have FEWER replicas if platforms are shared")
    print()
    
    print("3. STATISTICAL DISTRIBUTION:")
    print("   executecosimulation.py:")
    print("     - Does NOT support statistical distribution")
    print("     - Always uses per_server/per_client as-is")
    print()
    print("   simulation.py (lines 231-237, 293-298):")
    print("     - Supports statistical distribution via prewarm_config")
    print("     - Can override per_server/per_client per node")
    print("     - Uses sample_replica_count() to sample per-node replica counts")
    print()
    print("   ⚠️  DIFFERENCE: If prewarm_config has 'distribution': 'statistical',")
    print("      simulation.py will create different numbers of replicas per node.")
    print()
    
    print("4. PLATFORM ORDERING:")
    print("   executecosimulation.py (lines 619-622):")
    print("     suitable_platforms = [p for p in node_platforms[node_name]")
    print("                           if p['platform_type'] in supported_platforms]")
    print("     # Uses order from node_platforms dict (platform_id order)")
    print()
    print("   simulation.py (lines 239-243):")
    print("     suitable_platforms = [platform for platform in node.platforms.items")
    print("                           if (platform.type['shortName'] in supported_platforms")
    print("                           and (node, platform) not in assigned_platforms)]")
    print("     # Uses order from node.platforms.items (should be same, but depends on creation)")
    print()
    print("   ⚠️  POTENTIAL DIFFERENCE: If platform ordering differs, replicas will be")
    print("      created on different platforms even with same counts.")
    print()
    
    print("5. PLATFORM ID ASSIGNMENT:")
    print("   executecosimulation.py (lines 585-601):")
    print("     - Creates platform_id sequentially: platform_id = 0, 1, 2, ...")
    print("     - Assigns IDs in order: client nodes first, then server nodes")
    print("     - Order: client_node0, client_node1, ..., node0, node1, ...")
    print()
    print("   simulation.py (create_nodes, lines 118-132):")
    print("     - Creates platform_id sequentially: platform_id = 0, 1, 2, ...")
    print("     - Assigns IDs in order: nodes appear in infrastructure['nodes'] order")
    print("     - Order: depends on how nodes are created in prepare_simulation_config")
    print()
    print("   ⚠️  POTENTIAL DIFFERENCE: If node order differs, platform IDs will differ!")
    print("      - executecosimulation.py: Uses nodes from infrastructure_config['nodes']")
    print("      - simulation.py: Uses nodes from infrastructure['nodes']")
    print("      - Both should use same order if same infrastructure config is passed")
    print()
    
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("❌ NO - executecosimulation.py and simulation.py do NOT create the exact same replicas!")
    print()
    print("MAIN DISCREPANCIES:")
    print()
    print("1. PLATFORM ASSIGNMENT TRACKING (CRITICAL):")
    print("   - executecosimulation.py allows platforms to be shared across task types")
    print("   - simulation.py prevents double-booking (one replica per platform total)")
    print("   - This means simulation.py will have FEWER replicas if platforms are limited")
    print()
    print("2. STATISTICAL DISTRIBUTION (IF ENABLED):")
    print("   - simulation.py can use statistical distribution to vary replica counts")
    print("   - executecosimulation.py always uses fixed per_server/per_client")
    print()
    print("3. PLATFORM ORDERING (POTENTIAL):")
    print("   - If platform ordering differs, different platforms will be selected")
    print("   - Even if counts match, actual platforms may differ")
    print()
    print("RECOMMENDATION:")
    print("   To make them consistent, you need to:")
    print("   1. Add assigned_platforms tracking to executecosimulation.py")
    print("   2. Disable statistical distribution in simulation.py (or add to executecosimulation.py)")
    print("   3. Ensure platform ordering is deterministic and matches")
    print()

if __name__ == "__main__":
    analyze_replica_creation_differences()












