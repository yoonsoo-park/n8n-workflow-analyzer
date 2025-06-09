#!/usr/bin/env python3
"""
Test script for network analysis functionality.

This script tests the network analysis and graph processing capabilities
of the n8n workflow analyzer.
"""

import sys
import os
import json
from pathlib import Path
import networkx as nx

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import N8nWorkflow, WorkflowCollection
from parser import N8nWorkflowParser
from network.network_analyzer import NetworkAnalyzer
from config import get_config_manager


def test_workflow_graph_creation(collection):
    """Test creating graphs from individual workflows."""
    print("\nTesting workflow graph creation...")
    
    analyzer = NetworkAnalyzer()
    
    for workflow in collection.workflows:
        print(f"\nWorkflow: {workflow.name}")
        
        # Create graph
        graph = analyzer.create_workflow_graph(workflow)
        
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Edges: {len(graph.edges)}")
        print(f"  Node types: {set(graph.nodes[node]['type'] for node in graph.nodes)}")
        
        # Test graph properties
        if len(graph.nodes) > 0:
            print(f"  Is directed: {graph.is_directed()}")
            print(f"  Is connected: {len(list(nx.weakly_connected_components(graph))) == 1}")
    
    return True


def test_centrality_measures(collection):
    """Test centrality measure calculations."""
    print("\nTesting centrality measures...")
    
    analyzer = NetworkAnalyzer()
    
    for workflow in collection.workflows:
        print(f"\nWorkflow: {workflow.name}")
        
        # Create graph
        graph = analyzer.create_workflow_graph(workflow)
        
        if len(graph.nodes) < 2:
            print("  Skipping - not enough nodes for centrality analysis")
            continue
        
        # Calculate centralities
        centralities = analyzer.calculate_centrality_measures(graph)
        
        print(f"  Centrality measures calculated for {len(centralities)} nodes:")
        
        # Show top nodes by different measures
        measures = ['degree_centrality', 'betweenness_centrality', 'pagerank']
        for measure in measures:
            sorted_nodes = sorted(
                centralities.items(),
                key=lambda x: getattr(x[1], measure),
                reverse=True
            )
            if sorted_nodes:
                top_node = sorted_nodes[0]
                node_name = graph.nodes[top_node[0]].get('name', top_node[0])
                value = getattr(top_node[1], measure)
                print(f"    Top {measure}: {node_name} ({value:.3f})")
    
    return True


def test_community_detection(collection):
    """Test community detection functionality."""
    print("\nTesting community detection...")
    
    analyzer = NetworkAnalyzer()
    
    # Create collection graph
    collection_graph = analyzer.create_collection_graph(collection)
    
    print(f"Collection graph: {len(collection_graph.nodes)} workflows, {len(collection_graph.edges)} connections")
    
    # Detect communities
    communities = analyzer.detect_communities(collection_graph)
    
    print(f"Detected {len(communities)} communities:")
    for community in communities:
        print(f"  Community {community.community_id}: {community.size} workflows")
        print(f"    Workflows: {community.nodes}")
    
    return communities


def test_path_analysis(collection):
    """Test path analysis functionality."""
    print("\nTesting path analysis...")
    
    analyzer = NetworkAnalyzer()
    
    for workflow in collection.workflows:
        print(f"\nWorkflow: {workflow.name}")
        
        # Create graph
        graph = analyzer.create_workflow_graph(workflow)
        
        if len(graph.nodes) < 2:
            print("  Skipping - not enough nodes for path analysis")
            continue
        
        # Analyze paths
        path_analysis = analyzer.analyze_paths(graph)
        
        print(f"  Diameter: {path_analysis.diameter}")
        print(f"  Average path length: {path_analysis.average_path_length:.2f}")
        print(f"  Critical paths: {len(path_analysis.critical_paths)}")
        print(f"  Bottlenecks: {path_analysis.bottlenecks}")
        
        # Show critical paths
        for i, path in enumerate(path_analysis.critical_paths[:3]):
            node_names = [graph.nodes[node_id].get('name', node_id) for node_id in path]
            print(f"    Critical path {i+1}: {' -> '.join(node_names)}")
    
    return True


def test_graph_metrics(collection):
    """Test graph-level metrics calculation."""
    print("\nTesting graph metrics...")
    
    analyzer = NetworkAnalyzer()
    
    # Test individual workflow graphs
    for workflow in collection.workflows:
        print(f"\nWorkflow: {workflow.name}")
        
        graph = analyzer.create_workflow_graph(workflow)
        metrics = analyzer.calculate_graph_metrics(graph)
        
        print(f"  Nodes: {metrics['node_count']}")
        print(f"  Edges: {metrics['edge_count']}")
        print(f"  Density: {metrics['density']:.3f}")
        print(f"  Clustering coefficient: {metrics['clustering_coefficient']:.3f}")
        print(f"  Connected components: {metrics['connected_components']}")
        print(f"  Average degree: {metrics['average_degree']:.2f}")
    
    # Test collection graph
    print(f"\nCollection graph metrics:")
    collection_graph = analyzer.create_collection_graph(collection)
    collection_metrics = analyzer.calculate_graph_metrics(collection_graph)
    
    for key, value in collection_metrics.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} unique values")
        else:
            print(f"  {key}: {value}")
    
    return True


def test_influential_nodes(collection):
    """Test influential node identification."""
    print("\nTesting influential node identification...")
    
    analyzer = NetworkAnalyzer()
    
    for workflow in collection.workflows:
        print(f"\nWorkflow: {workflow.name}")
        
        graph = analyzer.create_workflow_graph(workflow)
        
        if len(graph.nodes) < 3:
            print("  Skipping - not enough nodes for influence analysis")
            continue
        
        influential = analyzer.find_influential_nodes(graph, top_k=3)
        
        for measure, nodes in influential.items():
            if nodes:
                node_names = [graph.nodes[node_id].get('name', node_id) for node_id in nodes]
                print(f"  Top {measure}: {', '.join(node_names)}")
    
    return True


def test_dependency_analysis(collection):
    """Test workflow dependency analysis."""
    print("\nTesting dependency analysis...")
    
    analyzer = NetworkAnalyzer()
    
    # Analyze dependencies
    dependency_analysis = analyzer.analyze_workflow_dependencies(collection)
    
    print(f"Dependency analysis results:")
    print(f"  Dependency graph: {len(dependency_analysis['dependency_graph'].nodes)} workflows")
    print(f"  Dependency chains: {len(dependency_analysis['dependency_chains'])}")
    print(f"  Isolated workflows: {dependency_analysis['isolated_workflows']}")
    print(f"  Hub workflows: {dependency_analysis['hub_workflows']}")
    
    # Show dependency chains
    for i, chain in enumerate(dependency_analysis['dependency_chains'][:3]):
        print(f"  Dependency chain {i+1}: {' -> '.join(chain)}")
    
    return dependency_analysis


def save_network_results(collection, communities, dependency_analysis):
    """Save network analysis results to JSON files."""
    print("\nSaving network analysis results...")
    
    results_dir = Path("data/reports/network")
    results_dir.mkdir(exist_ok=True)
    
    analyzer = NetworkAnalyzer()
    
    # Save individual workflow analyses
    workflow_analyses = []
    for workflow in collection.workflows:
        graph = analyzer.create_workflow_graph(workflow)
        
        analysis = {
            "workflow_id": workflow.id or workflow.name,
            "workflow_name": workflow.name,
            "graph_metrics": analyzer.calculate_graph_metrics(graph),
            "centralities": {},
            "path_analysis": {}
        }
        
        if len(graph.nodes) >= 2:
            # Calculate centralities
            centralities = analyzer.calculate_centrality_measures(graph)
            analysis["centralities"] = {
                node_id: cent.to_dict() for node_id, cent in centralities.items()
            }
            
            # Path analysis
            path_analysis = analyzer.analyze_paths(graph)
            analysis["path_analysis"] = {
                "diameter": path_analysis.diameter,
                "average_path_length": path_analysis.average_path_length,
                "critical_paths": path_analysis.critical_paths,
                "bottlenecks": path_analysis.bottlenecks
            }
        
        workflow_analyses.append(analysis)
    
    with open(results_dir / "workflow_network_analysis.json", 'w') as f:
        json.dump(workflow_analyses, f, indent=2, default=str)
    
    # Save community detection results
    communities_data = []
    for community in communities:
        communities_data.append({
            "community_id": community.community_id,
            "nodes": community.nodes,
            "size": community.size,
            "modularity": community.modularity
        })
    
    with open(results_dir / "communities.json", 'w') as f:
        json.dump(communities_data, f, indent=2)
    
    # Save dependency analysis
    dependency_data = {
        "dependency_chains": dependency_analysis['dependency_chains'],
        "isolated_workflows": dependency_analysis['isolated_workflows'],
        "hub_workflows": dependency_analysis['hub_workflows'],
        "dependency_metrics": dependency_analysis['dependency_metrics']
    }
    
    with open(results_dir / "dependency_analysis.json", 'w') as f:
        json.dump(dependency_data, f, indent=2, default=str)
    
    print(f"Network analysis results saved to {results_dir}")


def main():
    """Main test function."""
    print("n8n Workflow Network Analysis Test Suite")
    print("=" * 45)
    
    try:
        # Initialize configuration
        config_manager = get_config_manager()
        print(f"Configuration loaded successfully")
        
        # Parse workflows
        parser = N8nWorkflowParser()
        sample_dir = Path("data/sample_workflows")
        collection = parser.parse_workflow_collection(sample_dir)
        
        if collection.total_workflows == 0:
            print("No workflows found to analyze!")
            return False
        
        print(f"Loaded {collection.total_workflows} workflows for network analysis")
        
        # Test network analysis components
        test_workflow_graph_creation(collection)
        test_centrality_measures(collection)
        communities = test_community_detection(collection)
        test_path_analysis(collection)
        test_graph_metrics(collection)
        test_influential_nodes(collection)
        dependency_analysis = test_dependency_analysis(collection)
        
        # Save results
        save_network_results(collection, communities, dependency_analysis)
        
        print("\nAll network analysis tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

