#!/usr/bin/env python3
"""
Test script for the n8n workflow analyzer.

This script tests the workflow analysis engine with sample data.
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import N8nWorkflow, WorkflowCollection
from parser import N8nWorkflowParser
from analysis.workflow_analyzer import WorkflowAnalyzer
from config import get_config_manager


def test_workflow_parser():
    """Test the workflow parser with sample data."""
    print("Testing workflow parser...")
    
    parser = N8nWorkflowParser()
    sample_dir = Path("data/sample_workflows")
    
    if not sample_dir.exists():
        print(f"Sample directory {sample_dir} not found!")
        return False
    
    # Parse all workflows in the sample directory
    collection = parser.parse_workflow_collection(sample_dir)
    
    print(f"Parsed {collection.total_workflows} workflows")
    print(f"Total nodes: {collection.total_nodes}")
    print(f"Total connections: {collection.total_connections}")
    print(f"Unique node types: {len(collection.all_node_types)}")
    
    # Print parsing summary
    summary = parser.get_parsing_summary()
    if summary["error_count"] > 0:
        print(f"Parsing errors: {summary['error_count']}")
        for error in summary["errors"]:
            print(f"  - {error}")
    
    if summary["warning_count"] > 0:
        print(f"Parsing warnings: {summary['warning_count']}")
        for warning in summary["warnings"][:5]:  # Show first 5 warnings
            print(f"  - {warning}")
    
    return collection


def test_workflow_analyzer(collection):
    """Test the workflow analyzer with parsed workflows."""
    print("\nTesting workflow analyzer...")
    
    analyzer = WorkflowAnalyzer()
    
    for workflow in collection.workflows:
        print(f"\nAnalyzing workflow: {workflow.name}")
        
        # Perform analysis
        result = analyzer.analyze_workflow(workflow)
        
        # Print basic metrics
        if "structure" in result.metrics:
            structure = result.metrics["structure"]
            print(f"  Nodes: {structure.node_count}")
            print(f"  Connections: {structure.connection_count}")
            print(f"  Max depth: {structure.max_depth}")
            print(f"  Max width: {structure.max_width}")
            print(f"  Unique node types: {structure.unique_node_types}")
            print(f"  Has error handling: {structure.has_error_handling}")
        
        if "complexity" in result.metrics:
            complexity = result.metrics["complexity"]
            print(f"  Structural complexity: {complexity.structural_complexity:.2f}")
            print(f"  Cognitive complexity: {complexity.cognitive_complexity:.2f}")
            print(f"  Cyclomatic complexity: {complexity.cyclomatic_complexity}")
            print(f"  Maintainability index: {complexity.maintainability_index:.2f}")
        
        # Print recommendations
        if result.recommendations:
            print("  Recommendations:")
            for rec in result.recommendations[:3]:  # Show first 3 recommendations
                print(f"    - {rec}")
    
    return True


def test_node_type_analysis(collection):
    """Test node type analysis across all workflows."""
    print("\nTesting node type analysis...")
    
    analyzer = WorkflowAnalyzer()
    all_node_types = {}
    
    for workflow in collection.workflows:
        result = analyzer.analyze_workflow(workflow)
        
        if "node_analysis" in result.metrics:
            node_analysis = result.metrics["node_analysis"]
            
            # Aggregate node type counts
            for node_type, count in node_analysis["type_distribution"].items():
                all_node_types[node_type] = all_node_types.get(node_type, 0) + count
    
    print("Node type distribution across all workflows:")
    sorted_types = sorted(all_node_types.items(), key=lambda x: x[1], reverse=True)
    for node_type, count in sorted_types:
        print(f"  {node_type}: {count}")
    
    return True


def test_connection_analysis(collection):
    """Test connection pattern analysis."""
    print("\nTesting connection analysis...")
    
    analyzer = WorkflowAnalyzer()
    
    for workflow in collection.workflows:
        result = analyzer.analyze_workflow(workflow)
        
        if "connection_analysis" in result.metrics:
            conn_analysis = result.metrics["connection_analysis"]
            
            print(f"\nWorkflow: {workflow.name}")
            print(f"  Total connections: {conn_analysis['total_connections']}")
            print(f"  Entry points: {len(conn_analysis['entry_points'])}")
            print(f"  Exit points: {len(conn_analysis['exit_points'])}")
            print(f"  Bottlenecks: {len(conn_analysis['bottlenecks'])}")
            
            if conn_analysis["bottlenecks"]:
                print("  Top bottlenecks:")
                for node_id, conn_count in conn_analysis["bottlenecks"][:3]:
                    print(f"    {node_id}: {conn_count} incoming connections")
    
    return True


def save_analysis_results(collection):
    """Save analysis results to JSON files."""
    print("\nSaving analysis results...")
    
    analyzer = WorkflowAnalyzer()
    results_dir = Path("data/reports")
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for workflow in collection.workflows:
        result = analyzer.analyze_workflow(workflow)
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            "workflow_id": result.workflow_id,
            "workflow_name": result.workflow_name,
            "analysis_timestamp": result.analysis_timestamp.isoformat(),
            "metrics": {},
            "patterns": result.patterns,
            "recommendations": result.recommendations
        }
        
        # Convert metrics to serializable format
        for key, value in result.metrics.items():
            if hasattr(value, '__dict__'):
                result_dict["metrics"][key] = value.__dict__
            else:
                result_dict["metrics"][key] = value
        
        all_results.append(result_dict)
        
        # Save individual workflow result
        output_file = results_dir / f"{workflow.id or 'unknown'}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
    
    # Save combined results
    combined_file = results_dir / "all_workflows_analysis.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Analysis results saved to {results_dir}")
    return True


def main():
    """Main test function."""
    print("n8n Workflow Analyzer Test Suite")
    print("=" * 40)
    
    try:
        # Initialize configuration
        config_manager = get_config_manager()
        print(f"Configuration loaded successfully")
        
        # Test parser
        collection = test_workflow_parser()
        if not collection or collection.total_workflows == 0:
            print("No workflows found to analyze!")
            return False
        
        # Test analyzer
        test_workflow_analyzer(collection)
        
        # Test specific analysis components
        test_node_type_analysis(collection)
        test_connection_analysis(collection)
        
        # Save results
        save_analysis_results(collection)
        
        print("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

