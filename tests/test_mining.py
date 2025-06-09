#!/usr/bin/env python3
"""
Test script for pattern mining functionality.

This script tests the pattern mining and data mining capabilities
of the n8n workflow analyzer.
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import N8nWorkflow, WorkflowCollection
from parser import N8nWorkflowParser
from mining.pattern_miner import PatternMiner
from config import get_config_manager


def test_frequent_pattern_mining(collection):
    """Test frequent pattern mining functionality."""
    print("\nTesting frequent pattern mining...")
    
    miner = PatternMiner()
    
    # Mine frequent patterns
    patterns = miner.mine_frequent_patterns(collection)
    
    print(f"Found {len(patterns)} frequent patterns:")
    for i, pattern in enumerate(patterns[:10]):  # Show top 10
        print(f"  {i+1}. {pattern}")
        print(f"     Workflows: {pattern.workflows}")
    
    return patterns


def test_association_rules(patterns):
    """Test association rule generation."""
    print("\nTesting association rule generation...")
    
    miner = PatternMiner()
    
    # Generate association rules
    rules = miner.generate_association_rules(patterns)
    
    print(f"Generated {len(rules)} association rules:")
    for i, rule in enumerate(rules[:10]):  # Show top 10
        print(f"  {i+1}. {rule}")
    
    return rules


def test_sequential_patterns(collection):
    """Test sequential pattern mining."""
    print("\nTesting sequential pattern mining...")
    
    miner = PatternMiner()
    
    # Mine sequential patterns
    sequences = miner.mine_sequential_patterns(collection)
    
    print(f"Found {len(sequences)} sequential patterns:")
    for i, sequence in enumerate(sequences[:10]):  # Show top 10
        print(f"  {i+1}. {' -> '.join(sequence)}")
    
    return sequences


def test_workflow_clustering(collection):
    """Test workflow clustering functionality."""
    print("\nTesting workflow clustering...")
    
    miner = PatternMiner()
    
    # Cluster workflows
    clusters = miner.cluster_workflows(collection)
    
    print(f"Created {len(clusters)} workflow clusters:")
    for cluster in clusters:
        print(f"\nCluster {cluster.cluster_id} ({cluster.size} workflows):")
        print(f"  Workflows: {cluster.workflows}")
        
        if cluster.characteristics:
            print(f"  Characteristics:")
            for key, value in cluster.characteristics.items():
                if isinstance(value, dict):
                    print(f"    {key}: {dict(list(value.items())[:3])}")  # Show first 3 items
                else:
                    print(f"    {key}: {value}")
    
    return clusters


def test_workflow_templates(collection):
    """Test workflow template identification."""
    print("\nTesting workflow template identification...")
    
    miner = PatternMiner()
    
    # Find workflow templates
    templates = miner.find_workflow_templates(collection)
    
    print(f"Identified {len(templates)} workflow templates:")
    for template in templates:
        print(f"\nTemplate {template['template_id']}:")
        print(f"  Workflows: {template['workflow_count']}")
        print(f"  Common node types: {template['common_structure'].get('node_types', [])}")
        print(f"  Average nodes: {template['common_structure'].get('avg_nodes', 0):.1f}")
    
    return templates


def test_node_usage_analysis(collection):
    """Test node usage pattern analysis."""
    print("\nTesting node usage pattern analysis...")
    
    miner = PatternMiner()
    
    # Analyze node usage patterns
    analysis = miner.analyze_node_usage_patterns(collection)
    
    print("Node usage analysis results:")
    
    print(f"\nNode frequency (top 10):")
    sorted_freq = sorted(analysis["node_frequency"].items(), key=lambda x: x[1], reverse=True)
    for node_type, count in sorted_freq[:10]:
        print(f"  {node_type}: {count}")
    
    print(f"\nPopular nodes (used in >70% of workflows):")
    for node in analysis["popular_nodes"]:
        print(f"  {node}")
    
    print(f"\nRare nodes (used in <10% of workflows):")
    for node in analysis["rare_nodes"]:
        print(f"  {node}")
    
    print(f"\nCommon node combinations:")
    for combo, count in list(analysis["node_combinations"].items())[:5]:
        print(f"  {combo}: {count}")
    
    print(f"\nCommon node sequences:")
    for sequence, count in list(analysis["node_sequences"].items())[:5]:
        print(f"  {sequence}: {count}")
    
    print(f"\nNode categories:")
    for category, nodes in analysis["node_categories"].items():
        print(f"  {category}: {len(nodes)} types")
        for node in nodes[:3]:  # Show first 3
            print(f"    {node}")
    
    return analysis


def save_mining_results(patterns, rules, sequences, clusters, templates, analysis):
    """Save mining results to JSON files."""
    print("\nSaving mining results...")
    
    results_dir = Path("data/reports/mining")
    results_dir.mkdir(exist_ok=True)
    
    # Save frequent patterns
    patterns_data = []
    for pattern in patterns:
        patterns_data.append({
            "items": list(pattern.items),
            "support": pattern.support,
            "frequency": pattern.frequency,
            "workflows": pattern.workflows
        })
    
    with open(results_dir / "frequent_patterns.json", 'w') as f:
        json.dump(patterns_data, f, indent=2)
    
    # Save association rules
    rules_data = []
    for rule in rules:
        rules_data.append({
            "antecedent": list(rule.antecedent),
            "consequent": list(rule.consequent),
            "support": rule.support,
            "confidence": rule.confidence,
            "lift": rule.lift,
            "conviction": rule.conviction
        })
    
    with open(results_dir / "association_rules.json", 'w') as f:
        json.dump(rules_data, f, indent=2)
    
    # Save sequential patterns
    with open(results_dir / "sequential_patterns.json", 'w') as f:
        json.dump(sequences, f, indent=2)
    
    # Save clusters
    clusters_data = []
    for cluster in clusters:
        cluster_data = {
            "cluster_id": cluster.cluster_id,
            "workflows": cluster.workflows,
            "size": cluster.size,
            "characteristics": cluster.characteristics
        }
        if cluster.centroid is not None:
            cluster_data["centroid"] = cluster.centroid.tolist()
        clusters_data.append(cluster_data)
    
    with open(results_dir / "workflow_clusters.json", 'w') as f:
        json.dump(clusters_data, f, indent=2)
    
    # Save templates
    with open(results_dir / "workflow_templates.json", 'w') as f:
        json.dump(templates, f, indent=2)
    
    # Save node usage analysis
    with open(results_dir / "node_usage_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Mining results saved to {results_dir}")


def main():
    """Main test function."""
    print("n8n Workflow Pattern Mining Test Suite")
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
        
        print(f"Loaded {collection.total_workflows} workflows for pattern mining")
        
        # Test pattern mining components
        patterns = test_frequent_pattern_mining(collection)
        rules = test_association_rules(patterns)
        sequences = test_sequential_patterns(collection)
        clusters = test_workflow_clustering(collection)
        templates = test_workflow_templates(collection)
        analysis = test_node_usage_analysis(collection)
        
        # Save results
        save_mining_results(patterns, rules, sequences, clusters, templates, analysis)
        
        print("\nAll pattern mining tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

