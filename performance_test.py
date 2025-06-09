#!/usr/bin/env python3
"""
Performance testing for the n8n workflow analyzer.

This script tests the performance of the n8n workflow analyzer with
workflows of various sizes and complexities.

Usage:
    python3 performance_test.py
"""

import os
import sys
import json
import time
import logging
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import (
    N8nWorkflow, WorkflowNode, WorkflowConnection, 
    WorkflowSettings, WorkflowCollection
)
from src.parser import N8nWorkflowParser
from src.analysis.workflow_analyzer import WorkflowAnalyzer
from src.mining.pattern_miner import PatternMiner
from src.network.network_analyzer import NetworkAnalyzer
from src.visualization.visualization_manager import VisualizationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_test')

# Node types for sample workflows
NODE_TYPES = [
    "n8n-nodes-base.httpRequest",
    "n8n-nodes-base.set",
    "n8n-nodes-base.if",
    "n8n-nodes-base.switch",
    "n8n-nodes-base.function",
    "n8n-nodes-base.merge",
    "n8n-nodes-base.spreadsheetFile",
    "n8n-nodes-base.emailSend",
    "n8n-nodes-base.slack",
    "n8n-nodes-base.googleSheets",
    "n8n-nodes-base.webhook",
    "n8n-nodes-base.cron",
    "n8n-nodes-base.noOp",
    "n8n-nodes-base.wait",
    "n8n-nodes-base.filter",
    "n8n-nodes-base.code",
    "n8n-nodes-base.itemLists"
]

def generate_random_workflow(node_count, complexity="medium"):
    """Generate a random workflow with the specified number of nodes."""
    workflow_id = f"perf_workflow_{node_count}_{complexity}"
    workflow_name = f"Performance Test Workflow ({node_count} nodes, {complexity} complexity)"
    
    # Create nodes
    nodes = []
    for i in range(node_count):
        node_type = random.choice(NODE_TYPES)
        node_name = f"{node_type.split('.')[-1]}_{i+1}"
        
        node = WorkflowNode(
            id=f"node_{i+1}",
            name=node_name,
            type=node_type,
            position={
                "x": 100 + (i % 10) * 200,
                "y": 100 + (i // 10) * 150
            },
            parameters={
                "operation": random.choice(["create", "read", "update", "delete"]),
                "resource": random.choice(["user", "product", "order", "invoice"]),
                "options": {
                    "limit": random.randint(10, 100)
                }
            }
        )
        nodes.append(node)
    
    # Create connections based on complexity
    connections = []
    if complexity == "low":
        # Simple linear connections
        for i in range(node_count - 1):
            connection = WorkflowConnection(
                source_node=f"node_{i+1}",
                target_node=f"node_{i+2}"
            )
            connections.append(connection)
    elif complexity == "medium":
        # Moderate branching
        connection_count = min(node_count * 1.5, node_count * (node_count - 1) / 2)
        connection_count = int(connection_count)
        
        for i in range(node_count - 1):
            # Always connect to next node
            connection = WorkflowConnection(
                source_node=f"node_{i+1}",
                target_node=f"node_{i+2}"
            )
            connections.append(connection)
            
            # Add some random connections
            if len(connections) < connection_count:
                for _ in range(min(2, connection_count - len(connections))):
                    target = random.randint(i+2, node_count)
                    if target <= node_count:
                        connection = WorkflowConnection(
                            source_node=f"node_{i+1}",
                            target_node=f"node_{target}"
                        )
                        connections.append(connection)
    else:  # high
        # Complex connections with many branches
        connection_count = min(node_count * 2.5, node_count * (node_count - 1) / 2)
        connection_count = int(connection_count)
        
        while len(connections) < connection_count:
            source = random.randint(1, node_count - 1)
            target = random.randint(source + 1, node_count)
            
            connection = WorkflowConnection(
                source_node=f"node_{source}",
                target_node=f"node_{target}"
            )
            
            # Check if connection already exists
            if not any(c.source_node == connection.source_node and 
                      c.target_node == connection.target_node for c in connections):
                connections.append(connection)
    
    # Create workflow settings
    settings = WorkflowSettings(
        execution_order="sequential",
        timezone="UTC",
        save_execution_progress=True,
        save_manual_executions=True
    )
    
    # Create workflow
    workflow = N8nWorkflow(
        id=workflow_id,
        name=workflow_name,
        nodes=nodes,
        connections=connections,
        settings=settings,
        version=1,
        active=True
    )
    
    return workflow

def test_workflow_analysis_performance(node_counts, complexities):
    """Test the performance of workflow analysis with various workflow sizes."""
    # Initialize components
    analyzer = WorkflowAnalyzer()
    
    results = {}
    
    for complexity in complexities:
        results[complexity] = {
            'node_counts': node_counts,
            'analysis_times': []
        }
        
        for node_count in node_counts:
            # Generate workflow
            workflow = generate_random_workflow(node_count, complexity)
            
            # Measure analysis time
            start_time = time.time()
            analyzer.analyze_workflow(workflow)
            end_time = time.time()
            
            analysis_time = end_time - start_time
            results[complexity]['analysis_times'].append(analysis_time)
            
            logger.info(f"Analyzed workflow with {node_count} nodes and {complexity} complexity in {analysis_time:.4f} seconds")
    
    return results

def test_pattern_mining_performance(workflow_counts, node_count=20, complexity="medium"):
    """Test the performance of pattern mining with various collection sizes."""
    # Initialize components
    miner = PatternMiner()
    
    results = {
        'workflow_counts': workflow_counts,
        'mining_times': []
    }
    
    for workflow_count in workflow_counts:
        # Generate workflows
        workflows = [
            generate_random_workflow(node_count, complexity)
            for _ in range(workflow_count)
        ]
        
        # Create collection
        collection = WorkflowCollection(workflows=workflows)
        
        # Measure mining time
        start_time = time.time()
        patterns = miner.mine_frequent_patterns(collection)
        rules = miner.generate_association_rules(patterns)
        end_time = time.time()
        
        mining_time = end_time - start_time
        results['mining_times'].append(mining_time)
        
        logger.info(f"Mined patterns from {workflow_count} workflows in {mining_time:.4f} seconds")
    
    return results

def test_network_analysis_performance(workflow_counts, node_count=20, complexity="medium"):
    """Test the performance of network analysis with various collection sizes."""
    # Initialize components
    network_analyzer = NetworkAnalyzer()
    
    results = {
        'workflow_counts': workflow_counts,
        'analysis_times': []
    }
    
    for workflow_count in workflow_counts:
        # Generate workflows
        workflows = [
            generate_random_workflow(node_count, complexity)
            for _ in range(workflow_count)
        ]
        
        # Create collection
        collection = WorkflowCollection(workflows=workflows)
        
        # Measure analysis time
        start_time = time.time()
        graph = network_analyzer.create_collection_graph(collection)
        metrics = network_analyzer.calculate_network_metrics(graph)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        results['analysis_times'].append(analysis_time)
        
        logger.info(f"Analyzed network of {workflow_count} workflows in {analysis_time:.4f} seconds")
    
    return results

def test_visualization_performance(node_counts, complexity="medium"):
    """Test the performance of visualization generation with various workflow sizes."""
    # Initialize components
    viz_manager = VisualizationManager(output_dir="data/performance_test")
    
    results = {
        'node_counts': node_counts,
        'visualization_times': []
    }
    
    for node_count in node_counts:
        # Generate workflow
        workflow = generate_random_workflow(node_count, complexity)
        
        # Measure visualization time
        start_time = time.time()
        viz_manager.visualize_workflow(workflow)
        end_time = time.time()
        
        visualization_time = end_time - start_time
        results['visualization_times'].append(visualization_time)
        
        logger.info(f"Visualized workflow with {node_count} nodes in {visualization_time:.4f} seconds")
    
    return results

def plot_performance_results(results, output_dir):
    """Plot performance test results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot workflow analysis performance
    if 'workflow_analysis' in results:
        plt.figure(figsize=(10, 6))
        
        for complexity, data in results['workflow_analysis'].items():
            plt.plot(data['node_counts'], data['analysis_times'], marker='o', label=f"{complexity} complexity")
        
        plt.xlabel('Number of Nodes')
        plt.ylabel('Analysis Time (seconds)')
        plt.title('Workflow Analysis Performance')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(output_dir / "workflow_analysis_performance.png")
        plt.close()
    
    # Plot pattern mining performance
    if 'pattern_mining' in results:
        data = results['pattern_mining']
        
        plt.figure(figsize=(10, 6))
        plt.plot(data['workflow_counts'], data['mining_times'], marker='o')
        plt.xlabel('Number of Workflows')
        plt.ylabel('Mining Time (seconds)')
        plt.title('Pattern Mining Performance')
        plt.grid(True)
        
        plt.savefig(output_dir / "pattern_mining_performance.png")
        plt.close()
    
    # Plot network analysis performance
    if 'network_analysis' in results:
        data = results['network_analysis']
        
        plt.figure(figsize=(10, 6))
        plt.plot(data['workflow_counts'], data['analysis_times'], marker='o')
        plt.xlabel('Number of Workflows')
        plt.ylabel('Analysis Time (seconds)')
        plt.title('Network Analysis Performance')
        plt.grid(True)
        
        plt.savefig(output_dir / "network_analysis_performance.png")
        plt.close()
    
    # Plot visualization performance
    if 'visualization' in results:
        data = results['visualization']
        
        plt.figure(figsize=(10, 6))
        plt.plot(data['node_counts'], data['visualization_times'], marker='o')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Visualization Time (seconds)')
        plt.title('Visualization Performance')
        plt.grid(True)
        
        plt.savefig(output_dir / "visualization_performance.png")
        plt.close()
    
    # Create combined plot
    plt.figure(figsize=(12, 8))
    
    if 'workflow_analysis' in results:
        for complexity, data in results['workflow_analysis'].items():
            plt.plot(data['node_counts'], data['analysis_times'], marker='o', label=f"Analysis ({complexity})")
    
    if 'visualization' in results:
        data = results['visualization']
        plt.plot(data['node_counts'], data['visualization_times'], marker='s', label="Visualization")
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Performance Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_dir / "performance_comparison.png")
    plt.close()
    
    # Save results as JSON
    with open(output_dir / "performance_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x if not isinstance(x, np.ndarray) else x.tolist())

def main():
    """Main function to run performance tests."""
    # Create output directory
    output_dir = Path('data/performance_test')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define test parameters
    node_counts = [10, 20, 50, 100, 200, 500]
    workflow_counts = [5, 10, 20, 50, 100]
    complexities = ["low", "medium", "high"]
    
    # Run tests
    logger.info("Running workflow analysis performance tests...")
    workflow_analysis_results = test_workflow_analysis_performance(node_counts, complexities)
    
    logger.info("Running pattern mining performance tests...")
    pattern_mining_results = test_pattern_mining_performance(workflow_counts)
    
    logger.info("Running network analysis performance tests...")
    network_analysis_results = test_network_analysis_performance(workflow_counts)
    
    logger.info("Running visualization performance tests...")
    visualization_results = test_visualization_performance(node_counts)
    
    # Combine results
    results = {
        'workflow_analysis': workflow_analysis_results,
        'pattern_mining': pattern_mining_results,
        'network_analysis': network_analysis_results,
        'visualization': visualization_results
    }
    
    # Plot results
    logger.info("Plotting performance results...")
    plot_performance_results(results, output_dir)
    
    logger.info(f"Performance test results saved to {output_dir}")

if __name__ == '__main__':
    main()

